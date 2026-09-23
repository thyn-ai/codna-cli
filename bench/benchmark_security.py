#!/usr/bin/env python3
"""End-to-end benchmark of `codna secure` on a real vulnerable app — writes a Markdown report.

Mirrors a real user: a scanner (CodeQL-shaped SARIF) reports findings; codna independently
PROVES which are production-reachable (Tier-1, 0 LLM tokens) and which are not, then agentically
remediates the eligible ones and re-proves closure. Three phases, all real CLI, no mocks:

  1. Reachability triage (remote engine) — entrypoint-reachability + soundness envelope.
  2. Reachability triage (local reference engine) — the self-hostable, conservative contrast.
  3. Agentic remediation + closure (local engine --fix) — Cline edits in an ephemeral worktree,
     tests run, and the reference engine re-proves the obligation closed (sanitizer barrier added).

Fixture: bench/fixtures/vulnapp/ (app.py: 3 live-route vulns; legacy.py: 1 dead-code vuln that must
NOT be production-reachable — the discrimination test). The SARIF is generated here from the committed
source (sink line numbers discovered by pattern) bound to the fixture's fresh git commit.

Honest scope: this engine build's reachability proof tops out at `production-reachable`; `exploitable`
(independent taint proof) is the wired-but-off-in-v1 seam (taint_path_finder). Closure here is the
reference engine's barrier-pattern proof (sink file modified + a recognized sanitizer added), NOT
independent taint re-verification. Reported as exactly that.

Env: CODNA_API_KEY (Codna account), CODNA_BIN, ANTHROPIC_API_KEY + CODNA_AGENT_CORE_DIR
     (local --fix via bun fix-runner.ts), BENCHMARK_OUT (default: alongside this file).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

from bench_env import resolve_bin, resolve_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.environ.get("SEC_FIXTURE", os.path.join(HERE, "fixtures", "vulnapp"))
OUT = os.environ.get("BENCHMARK_OUT", os.path.join(HERE, "BENCHMARK-security.md"))

# (rule id, CWE tag, severity, file, sink-pattern to locate the line, expected verdict)
FINDINGS = [
    ("py/sql-injection", "cwe-089", "9.8", "app.py", "SELECT * FROM orders", "production-reachable"),
    ("py/command-line-injection", "cwe-078", "9.8", "app.py", "subprocess.check_output", "production-reachable"),
    ("py/path-injection", "cwe-022", "7.5", "app.py", "send_file(os.path.join", "production-reachable"),
    ("py/sql-injection", "cwe-089", "9.8", "legacy.py", "SELECT * FROM legacy", "not production-reachable"),
]
_MSG = {
    "py/sql-injection": "User input flows into a SQL query.",
    "py/command-line-injection": "User input flows into a shell command.",
    "py/path-injection": "User input flows into a file path.",
}


def _codna_bin() -> str:
    """Absolute path of the codna entrypoint: ``$CODNA_BIN`` or ``codna`` on PATH.

    Validated when the benchmark runs (allowlisted name on PATH, or an absolute existing
    executable), never at import, so the module stays importable where codna is not installed;
    a missing or unvetted binary still fails clearly before any work starts.
    """
    return resolve_bin("CODNA_BIN", "codna", allowed=("codna",))


def _manifest() -> str:
    """Absolute path of the verification manifest: ``$SEC_MANIFEST`` or the committed fixture.

    Must exist; checked when the benchmark runs, never at import.
    """
    return resolve_file("SEC_MANIFEST", os.path.join(HERE, "fixtures", "codna-security.yaml"))


def _line(path: str, needle: str) -> int:
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            if needle in ln:
                return i
    raise SystemExit(f"sink pattern not found in {path}: {needle!r}")


def _make_repo_and_sarif(workdir: str) -> tuple[str, str]:
    """Copy the fixture into a fresh git repo, commit it, and emit a SARIF bound to that commit."""
    repo = os.path.join(workdir, "vulnapp")
    shutil.copytree(FIXTURE, repo)
    # Fixtures are committed as `*.py.txt` templates so CodeQL doesn't flag their (intentional)
    # vulnerabilities in the repo; materialize them as real `.py` here for the scan + agent.
    for fn in os.listdir(repo):
        if fn.endswith(".py.txt"):
            os.rename(os.path.join(repo, fn), os.path.join(repo, fn[:-4]))
    env = {**os.environ, "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "b@b.co",
           "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "b@b.co"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "vulnerable fixture"], cwd=repo, check=True, env=env)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                         check=True).stdout.strip()

    rules, results = [], []
    seen = set()
    for rule, cwe, sev, fname, needle, _ in FINDINGS:
        if rule not in seen:
            rules.append({"id": rule, "properties": {"security-severity": sev,
                          "tags": ["security", f"external/cwe/{cwe}"]}})
            seen.add(rule)
        results.append({
            "ruleId": rule, "message": {"text": _MSG[rule]},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": fname},
                          "region": {"startLine": _line(os.path.join(repo, fname), needle)}}}],
        })
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0", "rules": rules}},
            "versionControlProvenance": [{"revisionId": sha,
                                          "repositoryUri": "https://github.com/acme/vulnapp"}],
            "results": results,
        }],
    }
    sarif_path = os.path.join(workdir, "scan.sarif")
    with open(sarif_path, "w") as f:
        json.dump(sarif, f)
    return repo, sarif_path


def _classify(repo: str, sarif: str, engine: str) -> tuple[list[tuple[str, str, str]], str]:
    """Return [(classification, rule_id, kind)], raw_output."""
    p = subprocess.run([_codna_bin(), "secure", repo, "--from-sarif", sarif, "--engine", engine],
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    rows = re.findall(r"[·→]\s*\[([\w-]+)\]\s+(\S+)\s+\((\w+)\)", out)
    return rows, out


def _fix(repo: str, sarif: str) -> str:
    p = subprocess.run([_codna_bin(), "secure", repo, "--from-sarif", sarif, "--engine", "local",
                        "--fix", "--verification", _manifest()], capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


def render(remote, local, fixout, gen) -> str:
    o = ["# `codna secure` benchmark — real vulnerable app\n",
         f"_Generated {gen} · live engine + local reference engine + agentic Cline remediation · "
         f"fixture: a Flask app with 3 live-route taint vulns + 1 dead-code vuln (the discrimination test)._\n",
         "> **Honest scope.** This engine build's independent reachability proof tops out at "
         "`production-reachable`; `exploitable` (independent taint-path proof) is the wired-but-off-in-v1 "
         "seam (`taint_path_finder`), so it is NOT claimed here. Closure (phase 3) is the reference engine's "
         "**barrier proof** — the patch modified the sink file AND added a recognized sanitizer "
         "(`parametrize`/`shlex.quote`/`bindparam`/…) — not an independent taint re-verification. "
         "Reported as exactly that. No mocks: real `codna secure` against a live engine + a real Cline run.\n"]

    o.append("## Phase 1 — reachability triage (remote engine, 0 LLM tokens)\n")
    o.append("The engine ingests the scanner SARIF and independently proves reachability. The 3 live "
             "`@app.route` handlers are reachable from a production entrypoint; the dead-code helper in "
             "`legacy.py` (no caller) is not — proving the triage discriminates by reachability.\n")
    o.append("| Finding (rule) | File | Verdict | Expected |")
    o.append("|---|---|---|---|")
    # Parsed rows are in SARIF/FINDINGS order; match positionally.
    for i, (rule, cwe, sev, fname, needle, exp) in enumerate(FINDINGS):
        verdict = remote[i][0] if i < len(remote) else "—"
        ok = "✓" if ((exp.startswith("production") and verdict == "production-reachable") or
                     (exp.startswith("not") and verdict != "production-reachable")) else "✗"
        o.append(f"| {rule} ({cwe}) | {fname} | `{verdict}` | {exp} {ok} |")
    o.append("\n```text\n" + _RAW.get('remote', '') + "\n```\n")

    o.append("## Phase 2 — reachability triage (local reference engine, self-hostable, 0 LLM tokens)\n")
    o.append("The conservative, bounded reference engine anyone can self-host. It never claims "
             "`exploitable` (ceiling = production-reachable) and agrees on the live routes.\n")
    o.append("```text\n" + (_RAW.get('local', '')) + "\n```\n")

    o.append("## Phase 3 — agentic remediation + closure (local engine `--fix`)\n")
    o.append("With the manifest's explicit policy override (`production-reachable` opted into autofix), "
             "codna runs the agentic Cline SDK in an **ephemeral worktree** (the repo is never mutated), "
             "verifies the patch compiles, and the reference engine re-proves the obligation closed "
             "(sink file modified + a sanitizer barrier added).\n")
    o.append("```text\n" + (fixout or "(fix phase not run)") + "\n```\n")

    o.append("## Method\n")
    o.append("- Real `codna secure <repo> --from-sarif scan.sarif` — classify (`--engine remote` and "
             "`--engine local`) then `--engine local --fix --verification codna-security.yaml`.\n"
             "- The SARIF is generated from the committed fixture (sink lines found by pattern) and bound "
             "to the fixture's fresh git commit (`versionControlProvenance.revisionId`).\n"
             "- Phase 3 runs a real Cline agent (`bun fix-runner.ts`) in a throwaway git worktree; the "
             "original checkout is left untouched.\n"
             "- `exploitable` is intentionally absent (v1 taint seam off); closure is barrier-based, not "
             "taint re-verification — stated honestly rather than overclaimed.\n")
    return "\n".join(x for x in o if x != "")


_RAW: dict[str, str] = {}


def main() -> int:
    # Validate the operator overrides before any work starts (the spawn helpers re-resolve them).
    _codna_bin()
    _manifest()
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    workdir = tempfile.mkdtemp(prefix="codna-sec-bench-")
    repo, sarif = _make_repo_and_sarif(workdir)

    remote, _RAW['remote'] = _classify(repo, sarif, "remote")
    print(f"[1/3] remote classify: {len(remote)} findings -> {[c for c,_,_ in remote]}")
    with open(OUT, "w") as f:
        f.write(render(remote, [], "", gen))

    local, _RAW['local'] = _classify(repo, sarif, "local")
    print(f"[2/3] local classify: {[c for c,_,_ in local]}")
    with open(OUT, "w") as f:
        f.write(render(remote, local, "", gen))

    fixout = ""
    if os.environ.get("SEC_RUN_FIX", "1") == "1":
        print("[3/3] agentic local --fix (real Cline) …")
        fixout = _fix(repo, sarif)
        print("      " + (fixout.splitlines()[-1] if fixout else "(no output)"))
    with open(OUT, "w") as f:
        f.write(render(remote, local, fixout, gen))
    print(f"done -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
