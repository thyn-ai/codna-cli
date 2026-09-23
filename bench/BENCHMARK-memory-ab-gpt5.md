# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-27 20:09 UTC · 5 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-flask

_repo: https://github.com/pallets/flask · issue: send_file returns the wrong content-type for .webp files_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | TestSendfile, _CollectErrors, __enter__, __exit__, __init__, | 306,568→5,769 (53× smaller) | 920 sym idx (9.7s) · 10 recalled | 11s | $0.103 | ok |
| Codna −Telys | gpt-5.4 | TestSendfile, send_file, test_send_file, test_static_file | 288,845→5,769 (50× smaller) | off (baseline) | 11s | $0.104 | ok |
| Cursor | composer-2.5-fast | src/flask/helpers.py | 66,390 | — | 111s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | src/flask/helpers.py, tests/test_helpers.py | 806,690 | — | 81s | $0.305 est | ok |

_**Memory A/B** (same issue, engine-behind): Telys recalled **10** related symbol(s); localization **CHANGED** vs −Telys; Δtime +0s · Δcost $-0.001._

## clone-rich

_repo: https://github.com/Textualize/rich · issue: Table column width is miscalculated when a cell contains wide (CJK) unicode characters_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | Column, cells, copy | 1,010,097→5,036 (201× smaller) | 1893 sym idx (33.3s) · 10 recalled | 11s | $0.090 | ok |
| Codna −Telys | gpt-5.4 | Column, Row, _Cell, cells, copy, flexible, test_init_append_ | 976,796→5,036 (194× smaller) | off (baseline) | 19s | $0.144 | ok |
| Cursor | composer-2.5-fast | — | 151,763 | — | 577s | Composer (subscription — no per-token rate) | success |
| Codex | gpt-5.4 | rich/table.py, tests/test_table.py | 2,270,156 | — | 168s | $0.608 est | ok |

_**Memory A/B** (same issue, engine-behind): Telys recalled **10** related symbol(s); localization **CHANGED** vs −Telys; Δtime -8s · Δcost $-0.054._

## clone-requests

_repo: https://github.com/psf/requests · issue: Session.send does not retry when the underlying connection times out; add a retry-on-timeout path_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | Session, SessionRedirectMixin, TestRequests, get_redirect_ta | 172,422→5,934 (29× smaller) | 726 sym idx (9.2s) · 10 recalled | 10s | $0.066 | ok |
| Codna −Telys | gpt-5.4 | Session, SessionRedirectMixin, TestRequests, get_redirect_ta | 156,993→5,934 (26× smaller) | off (baseline) | 14s | $0.109 | ok |
| Cursor | composer-2.5-fast | src/requests/sessions.py | 63,553 | — | 121s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | src/requests/sessions.py, tests/test_requests.py | 2,541,997 | — | 244s | $0.708 est | ok |

_**Memory A/B** (same issue, engine-behind): Telys recalled **10** related symbol(s); localization **unchanged** vs −Telys; Δtime -4s · Δcost $-0.043._

## clone-typer

_repo: https://github.com/tiangolo/typer · issue: a default value of an Enum option is not shown correctly in --help_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | _split_opt, _typer_param_setup_autocompletion_compat, hello_ | 446,000→5,618 (79× smaller) | 2143 sym idx (23.9s) · 10 recalled | 14s | $0.058 | ok |
| Codna −Telys | gpt-5.4 | _complete_visible_commands, _split_opt, _typer_param_setup_a | 397,058→5,618 (71× smaller) | off (baseline) | 28s | $0.112 | ok |
| Cursor | composer-2.5-fast | typer/core.py | 163,733 | — | 160s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | tests/test_types.py, typer/core.py | 2,119,121 | — | 130s | $0.581 est | ok |

_**Memory A/B** (same issue, engine-behind): Telys recalled **10** related symbol(s); localization **CHANGED** vs −Telys; Δtime -14s · Δcost $-0.054._

## clone-black

_repo: https://github.com/psf/black · issue: string normalization mishandles f-strings with nested quotes_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | cache_dir, event_loop | 843,998→5,230 (161× smaller) | 2812 sym idx (27.7s) · 10 recalled | 18s | $0.083 | ok |
| Codna −Telys | gpt-5.4 | FakeContext, __init__, event_loop, fix_multiline_docstring,  | 809,061→5,230 (155× smaller) | off (baseline) | 24s | $0.081 | ok |
| Cursor | composer-2.5-fast | src/black/strings.py | 73,282 | — | 283s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | src/black/linegen.py, tests/test_black.py | 11,366,935 | — | 570s | $2.399 est | ok |

_**Memory A/B** (same issue, engine-behind): Telys recalled **10** related symbol(s); localization **CHANGED** vs −Telys; Δtime -6s · Δcost $+0.002._

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.

# Security benchmark — vulnapp

_Generated 2026-06-27 20:10 UTC · target: bundled vulnapp fixture (3 live-route vulns + 1 dead-code) · Codna proves reachability (0 LLM); all three attempt the fix._

| Engine | Capability | Result | Time | Status |
|---|---|---|--:|---|
| Codna (secure) | reachability proof (0 LLM) | 3/4 production-reachable, 1 unreachable | 0s | ok |
| Cursor | agentic fix (no reachability signal) | 1 file(s), 1 sanitizer barrier(s) added | 25s | ok |
| Codex | agentic fix (no reachability signal) | 2 file(s), 1 sanitizer barrier(s) added | 107s | ok |

## Notes
- **Codna** ingests the scanner SARIF and PROVES which findings are production-reachable (0 LLM tokens) — telling real risk from dead code — which Cursor/Codex cannot. On the bundled fixture (3 live-route vulns + 1 dead-code helper) Codna should report 3 production-reachable, 1 unreachable.
- Cursor/Codex fix agentically with **no reachability signal** (they patch whatever they find); 'sanitizer barriers' counts recognized remediations (parameterized query / shlex.quote / path validation) in their diff.
- `exploitable` (independent taint proof) is Codna's wired-but-off-in-v1 seam; closure is a barrier check, not taint re-verification — stated honestly, not overclaimed.
