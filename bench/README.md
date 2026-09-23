# Codna fix benchmark suite

Reproducible, side-by-side **fix** benchmarks: plug in any public GitHub repo + an issue, pick which
engines to run (**Codna −Telys**, **Codna +Telys**, **Cline**, **Cursor**, optional **Codex**), get one comparison. Built so **anyone** can run it
and verify the numbers — no mocks, no shared infra; each engine runs the same issue headless on its own
throwaway checkout. The whole point is earning trust through results you can reproduce.

```bash
# one repo + issue, current enterprise comparison
python benchmark_suite.py --repo https://github.com/pallets/flask \
  --issue "send_file returns the wrong content-type for .webp files"

# N random curated public repos (reproducible with --seed); pick a subset of engines
python benchmark_suite.py --random 100 --engines codna-nomem,codna,cline,cursor --seed 7 --out report.md

# just compare against what you have installed (others auto-skip with a reason)
python benchmark_suite.py --repo URL --issue "…" --engines cline,cursor

# add fail-first + deterministic end-state verification
python benchmark_suite.py --repo URL --issue "…" \
  --precheck-cmd "pytest -q tests/test_bug.py" \
  --verify-cmd "pytest -q tests/test_bug.py" \
  --mode fix
```

## Modes — security runs by default
`--mode` is **`all` by default**: every run includes the **security** comparison (Codna's reachability
proof) alongside fix. Use `--mode fix` or `--mode security` to isolate one.

Codna's security headline is **reachability proof**: ingest a scanner SARIF and prove (0 LLM tokens)
which findings are production-reachable — telling real risk from dead code — which Cline/Cursor/Codex can't.
```bash
# bundled vulnerable fixture (3 live-route vulns + 1 dead-code) + auto-generated SARIF
python benchmark_suite.py --mode security --engines codna,cline,cursor

# your own vulnerable repo + your scanner's SARIF
python benchmark_suite.py --mode security --repo /path/to/repo --sarif scan.sarif
```
Codna runs the reachability triage; Cline/Cursor/Codex attempt an agentic fix of the same vulns (no
reachability signal). The report shows what Codna *proves* alongside what the others *patch*.

## What it measures
Per engine, per repo: the **localized** file/symbol, **input/output/cache/total tokens**, evidence
reduction, **wall-clock**, **cost** (where reported), optional fail-first **precheck** via
`--precheck-cmd`, optional final verification via `--verify-cmd`, derived **fix verified**, and
**status**. The headline is Codna's structural edge: it feeds the agent localized evidence (often
25–190× smaller than the repo) and runs the **Monte-Carlo govern gate** before finalizing — vs a raw
agent ingesting far more context.

## Setup (each engine is optional — missing ones are skipped with a noted reason)

**Codna** — runs engine-behind (local Algenta SDK localization + govern gate + Cline).
```bash
pip install codna
export CODNA_API_KEY=...                # your Codna key
```

**Cline** — public Cline CLI, no Codna engine.
```bash
export CLINE_PROVIDER=openai
export CLINE_MODEL=gpt-5.4
export CLINE_API_KEY="$OPENAI_API_KEY"
```

**Cursor** — the headless Cursor agent.
```bash
curl https://cursor.com/install -fsS | bash      # installs cursor-agent
export CURSOR_API_KEY=...               # from cursor.com dashboard  (or run: cursor-agent login)
```

**Codex** — OpenAI's Codex CLI.
```bash
brew install codex                      # or: npm i -g @openai/codex
codex login                             # or: export OPENAI_API_KEY=...
```

## Options
- `--repo URL|path` + `--issue "…"` — one specific target (`--ref` to pin a branch/tag).
- `--random N` [`--seed S`] — N random repos from the audited case pool: the 10 multilingual
  current cases plus tracked prior soak cases recovered from `cli/bench/soak/*.md`. The script fails
  fast if `N` exceeds the defined case count, so it cannot silently claim a larger run.
- `--engines codna-nomem,codna,cline,cursor,codex` — any subset.
- `--precheck-cmd "cmd args"` — optional deterministic command that must fail on a clean baseline
  before any engines run. If it passes or times out, that repo is skipped because the bug was not
  reproduced.
- `--verify-cmd "cmd args"` — optional deterministic command run after each engine that applies a
  local patch. The command is tokenized with shell-style quoting but is not run through a shell.
- `--out report.md` — write the Markdown report (default: stdout). Progress goes to stderr.
- Per-engine overrides via env: `CODNA_BIN`/`CLINE_BIN`/`CURSOR_BIN`/`CODEX_BIN`,
  `CLINE_MODEL`/`CURSOR_MODEL`/`CODEX_MODEL`,
  `*_TIMEOUT_S`.

## Honest scope
Codna runs **inspect mode** (it localizes + plans + returns a patch ref; it does not apply to your
tree). Cline/Cursor/Codex are agentic and edit a **throwaway copy** (your repo is never modified).
Codna reports planner input/output/cache tokens plus reduced evidence tokens; Cline/Cursor/Codex report
their own agent consumption when their CLIs expose it. `Fix verified` is `verified` only when
`--precheck-cmd` fails on the baseline and `--verify-cmd` passes after the patch. A passing final
verify without fail-first proof is reported as `unqualified`. Codna engine-behind inspect mode returns
a patch ref instead of applying to the checkout, so its final verification remains `n/a` unless running
the local apply-capable mode; this prevents false accuracy claims.

The recovered soak cases are reproducible throughput/localization cases. Some carry legacy generic
bug prompts from the earlier soak; use verified failure fixtures before claiming fix accuracy.
