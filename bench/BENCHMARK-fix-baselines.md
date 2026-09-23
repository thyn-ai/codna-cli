# Baseline comparison — raw frontier models vs codna (fix localization)

_Generated 2026-06-27 15:46 UTC · GPT-5.4 (OpenAI) + Gemini-2.5-pro fix the SAME issues **separately, not through codna's infra** — direct API calls with the repo source pasted in (cap 480k chars ≈ 120k tokens). codna's Cline agent is Anthropic-only; these are independent baselines._

> The contrast is **context efficiency + localization**: codna reduces a 100k–1M-token repo to ~5k tokens of evidence and patches the right symbol; a raw model must ingest the whole repo (and large repos may not even fit). Raw-model patch *quality* is not scored here — this isolates codna's structural advantage.

## Per-repo: codna vs raw models

| Repo | Engine | Localized file | Context tokens in | Out tokens | Time | Status |
|---|---|---|--:|--:|--:|---|
| requests | **codna** (Sonnet+localize) | HTTPAdapter | ~5,934 | — | 108s | ok |
| requests | gpt-5.4 (raw) | src/requests/adapters.py | 54,092 | 268 | 2s | ok |
| requests | gemini-2.5-pro (raw) | src/requests/adapters.py | 62,999 | 167 | 125s | ok |
| click | **codna** (Sonnet+localize) | _NamedTextIOWrapper | ~5,261 | — | 2192s | ok |
| click | gpt-5.4 (raw) | src/click/core.py | 103,032 | 348 | 4s | ok |
| click | gemini-2.5-pro (raw) | src/click/core.py | 120,684 | 231 | 128s | ok |
| flask | **codna** (Sonnet+localize) | get_debug_flag | ~5,769 | — | 40s | ok |
| flask | gpt-5.4 (raw) | src/flask/helpers.py | 81,558 | 230 | 3s | ok |
| flask | gemini-2.5-pro (raw) | src/flask/helpers.py | 94,868 | 245 | 93s | ok |
| httpx | **codna** (Sonnet+localize) | — | ~5,000 | — | 2409s | exit 1 |
| httpx | gpt-5.4 (raw) | httpx/_models.py | 64,393 | 552 | 4s | ok |
| httpx | gemini-2.5-pro (raw) | httpx/_client.py | 77,248 | 315 | 155s | ok |
| rich | **codna** (Sonnet+localize) | get_character_cell_size | ~5,036 | — | 38s | ok |
| rich | gpt-5.4 (raw) | rich/table.py ⚠trunc | 139,830 | 22 | 4s | ok |
| rich | gemini-2.5-pro (raw) | rich/cells.py ⚠trunc | 160,276 | 160 | 58s | ok |

## Method
- Raw models get the repo's non-test `.py` source concatenated (bounded), the issue, and are asked for `{file, symbol, patch}` JSON in ONE call — no browsing, no agentic loop, no codna infra.
- `Context tokens in` is what each engine fed the model: codna's reduced evidence (~5k) vs the raw repo dump (the model's actual prompt_tokens). `⚠trunc` = the repo exceeded the context cap (codna's localization is *required* at that size).
- Direct OpenAI `chat/completions` + Gemini `generateContent`.
