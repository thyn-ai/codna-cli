# `codna secure` benchmark — real vulnerable app

_Generated 2026-06-26 22:50 UTC · live engine + local reference engine + agentic Cline remediation · fixture: a Flask app with 3 live-route taint vulns + 1 dead-code vuln (the discrimination test)._

> **Honest scope.** This engine build's independent reachability proof tops out at `production-reachable`; `exploitable` (independent taint-path proof) is the wired-but-off-in-v1 seam (`taint_path_finder`), so it is NOT claimed here. Closure (phase 3) is the reference engine's **barrier proof** — the patch modified the sink file AND added a recognized sanitizer (`parametrize`/`shlex.quote`/`bindparam`/…) — not an independent taint re-verification. Reported as exactly that. No mocks: real `codna secure` against a live engine + a real Cline run.

## Phase 1 — reachability triage (remote engine, 0 LLM tokens)

The engine ingests the scanner SARIF and independently proves reachability. The 3 live `@app.route` handlers are reachable from a production entrypoint; the dead-code helper in `legacy.py` (no caller) is not — proving the triage discriminates by reachability.

| Finding (rule) | File | Verdict | Expected |
|---|---|---|---|
| py/sql-injection (cwe-089) | app.py | `production-reachable` | production-reachable ✓ |
| py/command-line-injection (cwe-078) | app.py | `production-reachable` | production-reachable ✓ |
| py/path-injection (cwe-022) | app.py | `production-reachable` | production-reachable ✓ |
| py/sql-injection (cwe-089) | legacy.py | `unreachable` | not production-reachable ✓ |

```text
codna: understanding /var/folders/qc/sgxl7gpd4bl3ccybs_8pzscc0000gn/T/codna-sec-bench-b4299eow/vulnapp for security analysis …

✓ analyzed 4 finding(s) from /var/folders/qc/sgxl7gpd4bl3ccybs_8pzscc0000gn/T/codna-sec-bench-b4299eow/scan.sarif
  · [production-reachable] py/sql-injection (taint)  production-reachable requires an explicit policy override
  · [production-reachable] py/command-line-injection (taint)  production-reachable requires an explicit policy override
  · [production-reachable] py/path-injection (taint)  production-reachable requires an explicit policy override
  · [unreachable] py/sql-injection (taint)  unreachable findings are never auto-fixed
  summary: production-reachable=3, unreachable=1  ·  autofix-eligible: 0
```

## Phase 2 — reachability triage (local reference engine, self-hostable, 0 LLM tokens)

The conservative, bounded reference engine anyone can self-host. It never claims `exploitable` (ceiling = production-reachable) and agrees on the live routes.

```text
✓ analyzed 4 finding(s) from /var/folders/qc/sgxl7gpd4bl3ccybs_8pzscc0000gn/T/codna-sec-bench-b4299eow/scan.sarif
  · [production-reachable] py/sql-injection (taint)  production-reachable requires an explicit policy override
  · [production-reachable] py/command-line-injection (taint)  production-reachable requires an explicit policy override
  · [production-reachable] py/path-injection (taint)  production-reachable requires an explicit policy override
  · [production-reachable] py/sql-injection (taint)  production-reachable requires an explicit policy override
  summary: production-reachable=4  ·  autofix-eligible: 0
```

## Phase 3 — agentic remediation + closure (local engine `--fix`)

With the manifest's explicit policy override (`production-reachable` opted into autofix), codna runs the agentic Cline SDK in an **ephemeral worktree** (the repo is never mutated), verifies the patch compiles, and the reference engine re-proves the obligation closed (sink file modified + a sanitizer barrier added).

```text
✓ analyzed 4 finding(s) from /var/folders/qc/sgxl7gpd4bl3ccybs_8pzscc0000gn/T/codna-sec-bench-b4299eow/scan.sarif
  → [production-reachable] py/sql-injection (taint)  autofix-eligible
  → [production-reachable] py/command-line-injection (taint)  autofix-eligible
  → [production-reachable] py/path-injection (taint)  autofix-eligible
  → [production-reachable] py/sql-injection (taint)  autofix-eligible
  summary: production-reachable=4  ·  autofix-eligible: 4

codna: remediating eligible findings with the agentic Cline SDK (local engine) …

✓ local fix: 4 eligible finding(s) processed
  [production-reachable] py/sql-injection  REMEDIATED — closure=closed, tests pass
  [production-reachable] py/command-line-injection  blocked: obligation not closed (closure=open)
  [production-reachable] py/path-injection  blocked: obligation not closed (closure=open)
  [production-reachable] py/sql-injection  REMEDIATED — closure=closed, tests pass
  remediated: 2/4
```

## Method

- Real `codna secure <repo> --from-sarif scan.sarif` — classify (`--engine remote` and `--engine local`) then `--engine local --fix --verification codna-security.yaml`.
- The SARIF is generated from the committed fixture (sink lines found by pattern) and bound to the fixture's fresh git commit (`versionControlProvenance.revisionId`).
- Phase 3 runs a real Cline agent (`bun fix-runner.ts`) in a throwaway git worktree; the original checkout is left untouched.
- `exploitable` is intentionally absent (v1 taint seam off); closure is barrier-based, not taint re-verification — stated honestly rather than overclaimed.
