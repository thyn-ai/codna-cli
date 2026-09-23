#!/usr/bin/env python3
"""Codna vs Cursor vs Cline — one reusable, side-by-side fix benchmark on public repos.

Plug in a GitHub URL (or local path) + an issue, pick the engines, get a comparison. Each selected
engine runs the SAME issue headless on its own (no shared infra) in an isolated checkout, and the
suite reports localization, context/tokens, wall-clock, and status side by side.

  codna   `codna fix <repo> --issue … --json`  (engine-behind: Algenta localization + Monte-Carlo
                                                 govern gate + Cline; needs the engine+sidecar up)
  cline   `cline --json --auto-approve …`       (public Cline CLI; no Codna engine)
  cursor  `cursor-agent -p … --force`           (Cursor's agent; needs CURSOR_API_KEY)
  codex   `codex exec … --full-auto`            (optional OpenAI Codex CLI)

Usage:
  python benchmark_suite.py --repo https://github.com/pallets/flask --issue "send_file webp type"
  python benchmark_suite.py --repo /path/to/repo --issue "…" --engines codna-nomem,codna,cline,cursor
  python benchmark_suite.py --repo URL --issue "…" --out report.md --ref main

Env: codna -> CODNA_API_KEY (+ CODNA_BIN); cline -> CLINE_*;
     cursor -> CURSOR_API_KEY (+ CURSOR_BIN); codex -> codex login / OPENAI_API_KEY (+ CODEX_BIN).
     Missing creds for an engine = skipped with a noted reason, not a crash.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import threading
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from benchmark_cases import benchmark_cases
from benchmark_validation import (
    attach_precheck as _attach_precheck,
    attach_verification as _validation_attach_verification,
    parse_precheck_cmd as _parse_precheck_cmd,
    parse_verify_cmd as _parse_verify_cmd,
    run_precheck as _validation_run_precheck,
    verify_unavailable as _verify_unavailable,
)

CODNA = os.environ.get("CODNA_BIN", "codna")
CURSOR = os.environ.get("CURSOR_BIN", os.path.expanduser("~/.local/bin/cursor-agent"))
CODEX = os.environ.get("CODEX_BIN", "codex")
CLINE = os.environ.get("CLINE_BIN", "cline")
# Codna's LOCAL agent head: codna spawns the vendored Cline (`bun algenta/fix-runner.ts`) itself — no
# external engine/sidecar to run. Point at agent-core/vendor/cline (override with CODNA_AGENT_CORE_DIR).
AGENT_CORE = os.environ.get("CODNA_AGENT_CORE_DIR",
                            os.path.expanduser("~/Developer/codna/agent-core/vendor/cline"))
AGENT_CORE_URL = os.environ.get("AGENT_CORE_URL", "http://127.0.0.1:18601")  # WARM sidecar (run-server.ts)
_SIDECAR_LOCK = threading.Lock()
_SIDECAR = {}  # holds the long-lived sidecar Popen once started — started ONCE, reused for every repo


def _sidecar_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(AGENT_CORE_URL + "/health", timeout=2)
        return True
    except Exception:
        return False


def _ensure_sidecar() -> bool:
    """Start the agent-core sidecar ONCE on a fixed port and keep it HOT; reuse it for every fix
    (no cold `bun` spawn per repo). Idempotent + thread-safe."""
    if _sidecar_up():
        return True
    with _SIDECAR_LOCK:
        if _sidecar_up():
            return True
        port = AGENT_CORE_URL.rsplit(":", 1)[-1]
        if not _which("bun") or not os.path.exists(os.path.join(AGENT_CORE, "algenta", "run-server.ts")):
            return False
        env = dict(os.environ, AGENT_CORE_PORT=port)
        _SIDECAR["proc"] = subprocess.Popen(["bun", "algenta/run-server.ts"], cwd=AGENT_CORE, env=env,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            if _sidecar_up():
                return True
            time.sleep(0.5)
        return False
FIX_PROMPT = ("Fix this bug with the minimal code change. Do not ask questions — locate the "
              "responsible function and make the edit. BUG: {issue}")

CURATED = benchmark_cases()
DEFAULT_ENGINES = "codna-nomem,codna,cline,cursor"
ENGINE_CHOICES = ("codna-nomem", "codna", "cline", "cursor", "codex")


def _run(cmd, cwd=None, timeout=1800, env=None, stdin=None):
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, stdin=stdin)
        return p.returncode, (p.stdout or ""), (p.stderr or ""), time.perf_counter() - t
    except subprocess.TimeoutExpired:
        return 124, "", "timeout", time.perf_counter() - t
    except OSError as exc:
        return 127, "", str(exc), time.perf_counter() - t


def _run_precheck(repo: str, precheck_cmd: list[str] | None) -> dict:
    return _validation_run_precheck(
        repo,
        precheck_cmd,
        run=_run,
        timeout=int(os.environ.get("BENCH_PRECHECK_TIMEOUT_S", "300")),
    )


def _attach_verification(row: dict, engine: str, copydir: str, verify_cmd: list[str] | None) -> dict:
    return _validation_attach_verification(
        row,
        engine,
        copydir,
        verify_cmd,
        run=_run,
        bench_mode=os.environ.get("CODNA_BENCH_MODE"),
        timeout=int(os.environ.get("BENCH_VERIFY_TIMEOUT_S", "300")),
    )


# Cursor-agent and Codex report TOKENS in detail but NO USD cost (verified empirically — neither CLI nor
# the underlying OpenAI/Anthropic APIs emit dollars; cost is always derived from tokens × the plan rate).
# Codna's own cost is REAL (engine planner usage). For the others we apply the SAME arithmetic every billing
# dashboard does — published per-1M-token rates — clearly marked. Override: CODNA_BENCH_RATES_JSON.
_DEFAULT_RATES = {  # USD per 1M tokens; "cached" = cache-read input rate
    "gpt-5.4": {"in": 1.25, "cached": 0.125, "out": 10.0},   # OpenAI GPT-5-class (override via env)
    "claude-sonnet-4-6": {"in": 3.0, "cached": 0.30, "out": 15.0},
}


def _rates():
    r = {k: dict(v) for k, v in _DEFAULT_RATES.items()}
    try:
        r.update(json.loads(os.environ.get("CODNA_BENCH_RATES_JSON") or "{}"))
    except (ValueError, TypeError):
        pass
    return r


def _token_cost(model, *, input_tokens, output_tokens, cached_tokens=0):
    """Derive USD from REPORTED tokens × published per-1M rate (cached input billed at the cache rate).
    Returns (cost_usd|None, note). None when the model has no known rate (e.g. Cursor's Composer)."""
    rate = _rates().get(model)
    if not rate or input_tokens is None:
        return None, ""
    non_cached = max((input_tokens or 0) - (cached_tokens or 0), 0)
    cr = rate.get("cached", rate["in"])
    cost = (non_cached * rate["in"] + (cached_tokens or 0) * cr + (output_tokens or 0) * rate["out"]) / 1_000_000
    return round(cost, 4), f"{model} @ ${rate['in']}/${rate['out']}/M"


def _first_number(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def _sum_tokens(input_tokens, output_tokens):
    if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        return input_tokens + output_tokens
    return None


def _parse_engines(raw: str) -> list[str]:
    engines: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        engine = item.strip()
        if not engine:
            continue
        if engine not in RUNNERS:
            unknown.append(engine)
            continue
        if engine not in seen:
            engines.append(engine)
            seen.add(engine)
    if unknown:
        raise ValueError(
            "unknown engine(s): "
            + ", ".join(unknown)
            + " (choose from: "
            + ", ".join(ENGINE_CHOICES)
            + ")"
        )
    if not engines:
        raise ValueError("no valid engines (choose from: " + ", ".join(ENGINE_CHOICES) + ")")
    return engines


def _select_random_jobs(count: int, seed: int | None) -> list[tuple[str, str]]:
    if count < 0:
        raise ValueError("--random must be non-negative")
    if count > len(CURATED):
        raise ValueError(
            f"--random {count} requested but only {len(CURATED)} curated benchmark cases are defined; "
            "add more cases before claiming a larger run"
        )
    rng = random.Random(seed) if seed is not None else random
    return rng.sample(CURATED, count)


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout


def _which(path_or_cmd):
    return shutil.which(path_or_cmd) or (os.path.exists(path_or_cmd) and path_or_cmd) or None


def _engine_url_configured() -> bool:
    return any(os.environ.get(key) for key in ("CODNA_ENGINE_URL", "ALGENTA_ENGINE_URL", "ALGENTA_BASE_URL"))


def _ensure_codna_runtime() -> str | None:
    if _engine_url_configured():
        return None
    rc, _out, err, _dt = _run([CODNA, "doctor", "--start-stack"], timeout=120)
    if rc != 0:
        last = err.strip().splitlines()
        return last[-1][:120] if last else "codna doctor --start-stack failed"
    return None


# ── per-engine runners → common result dict ────────────────────────────────────────────────────
# Only ONE Codna engine-behind agentic fix runs at a time: there is a single local engine + sidecar, so
# two concurrent runs would contend (and the +Telys vs −Telys A/B needs clean, uncontended wall-clock).
# Cline/Cursor/Codex run their own processes and stay parallel to whichever Codna run holds the lock.
_ENGINE_BEHIND_LOCK = threading.Lock()


def _codna_fix(repo, issue, *, use_memory):
    """Codna engine-behind, inspect mode, JSON. Reads the repo in place (snapshot is server-side).

    A/B knob: ``use_memory`` → index the repo with Telys (real semantic embedder, key-gated /v1/embeddings)
    and run ``codna fix --memory on`` so recalled ``related_symbols`` feed the engine; else ``--memory off``
    (identical run, no memory) — the controlled baseline. Telys unavailable is SURFACED in the Memory
    column (never silently dropped), and the fix still runs so the row is comparable."""
    if not os.environ.get("CODNA_API_KEY"):
        return {"status": "skipped", "notes": "CODNA_API_KEY unset (engine not configured)"}
    runtime_error = _ensure_codna_runtime()
    if runtime_error:
        return {"status": "skipped", "notes": runtime_error}
    mem, memflag, idt = "off (baseline)", "off", 0.0
    if use_memory:
        irc, iout, ierr, idt = _run([CODNA, "memory", "index", repo], timeout=900)   # no --language → AUTO-DETECT
        msym = re.search(r"indexed (\d+) symbols", iout)
        if irc == 0 and msym:
            mem, memflag = f"{msym.group(1)} sym idx ({idt:.1f}s)", "on"
        else:  # surfaced, never silent — kernel/extra/engine missing, unparsed repo, etc.
            last = (ierr or iout).strip().splitlines()
            mem, memflag = "unavailable: " + (last[-1][:48] if last else f"rc={irc}"), "off"
    with _ENGINE_BEHIND_LOCK:  # one agentic engine-behind run at a time → clean A/B timing
        rc, out, err, dt = _run([CODNA, "fix", repo, "--issue", issue, "--json", "--memory", memflag],
                                timeout=int(os.environ.get("CODNA_TIMEOUT_S", "2400")))
    m = re.search(r"\{.*\}", out, re.DOTALL)
    d = json.loads(m.group(0)) if m else {}
    ctx = d.get("context") or {}
    ratio = ctx.get("reduction_ratio")
    related = d.get("related_symbols") or []
    usage = d.get("planner_usage") or {}
    in_t = _first_number(usage, "input_tokens", "inputTokens")
    out_t = _first_number(usage, "output_tokens", "outputTokens")
    cache_r = _first_number(usage, "cache_read_tokens", "cached_input_tokens", "cacheReadTokens") or 0
    total_t = _first_number(usage, "total_tokens", "totalTokens") or _sum_tokens(in_t, out_t)
    if memflag == "on":
        mem += f" · {len(related)} recalled" if related else " · 0 recalled"
    return {
        "engine": "Codna +Telys" if use_memory else "Codna −Telys",
        "model": d.get("runtime_model") or "claude-sonnet-4-6",
        "localized": ", ".join(d.get("impacted_symbols") or []) or "—",
        "context_in": in_t,
        "out_tokens": out_t,
        "cache_read": cache_r,
        "total_tokens": total_t,
        "evidence_tokens": ctx.get("evidence_bundle_tokens"),
        "context_note": f"{ctx.get('raw_token_estimate'):,}→{ctx.get('evidence_bundle_tokens'):,} ({ratio:.0f}× smaller)"
                        if ratio and ctx.get("raw_token_estimate") else None,
        "cost_usd": d.get("cost_usd"), "time_s": round(dt), "index_s": round(idt, 1),
        "patch": d.get("patch_ref"), "memory": mem, "related_symbols": related,
        "status": "ok" if d.get("patch_ref") else (f"exit {rc}" if rc else "no-patch"),
        "notes": d.get("root_cause", "")[:60] if d else (err.strip().splitlines()[-1][:80] if err.strip() else ""),
    }


def run_codna_local(repo, issue, copydir):
    """Codna LOCAL execution head (NO engine — opt-in via CODNA_BENCH_MODE=local): a WARM agent-core
    sidecar (run-server.ts, started ONCE and reused) runs the vendored-Cline fix via POST /run. Fast, but
    NO localization / Monte-Carlo govern / reranking — it is NOT the product comparison."""
    import urllib.request
    if not _ensure_sidecar():
        return {"status": "skipped", "notes": f"agent-core sidecar not up ({AGENT_CORE_URL}); need bun + run-server.ts"}
    _git(copydir, "stash", "-u")
    provider = os.environ.get("CODNA_FIX_PROVIDER", "openai")
    model = os.environ.get("CODNA_FIX_MODEL", "gpt-5.4")
    payload = {"working_dir": copydir, "prompt": FIX_PROMPT.format(issue=issue),
               "provider": provider, "model": model, "channel": "cli",
               "limits": {"maxIterations": int(os.environ.get("CODNA_MAX_ITER", "20"))}}
    pk = os.environ.get("OPENAI_API_KEY") if provider == "openai" else os.environ.get("ANTHROPIC_API_KEY")
    if pk:
        payload["provider_api_key"] = pk
    headers = {"Content-Type": "application/json"}
    if os.environ.get("AGENT_CORE_AUTH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['AGENT_CORE_AUTH_TOKEN']}"
    t = time.perf_counter()
    try:
        req = urllib.request.Request(AGENT_CORE_URL + "/run", data=json.dumps(payload).encode(), headers=headers)
        body = urllib.request.urlopen(req, timeout=int(os.environ.get("CODNA_TIMEOUT_S", "600"))).read().decode()
    except Exception as exc:  # noqa: BLE001 — surface sidecar/network failure as a row, not a crash
        return {"engine": "Codna", "model": model, "time_s": round(time.perf_counter() - t),
                "status": "error", "notes": str(exc)[:70]}
    dt = time.perf_counter() - t
    final = {}
    for line in body.splitlines():                       # NDJSON; the "final" frame carries telemetry
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "final":
            final = ev
    tel = final.get("telemetry", {}) or {}
    in_t, out_t, cache_r = tel.get("tokens_in_uncached"), tel.get("tokens_out"), tel.get("cache_read_tokens", 0)
    cost = tel.get("total_cost") or None
    if not cost:
        cost, _ = _token_cost(model, input_tokens=in_t, output_tokens=out_t, cached_tokens=cache_r)
    changed = final.get("files_changed") or [ln for ln in _git(copydir, "diff", "--name-only").splitlines()]
    status = final.get("status")
    return {"engine": "Codna", "model": model, "localized": ", ".join(changed) or "—",
            "context_in": in_t, "out_tokens": out_t, "cache_read": cache_r,
            "total_tokens": (in_t + out_t) if (in_t is not None and out_t is not None) else None,
            "cost_usd": cost, "cost_note": f"Codna warm sidecar ({provider}/{model})", "time_s": round(dt),
            "status": "ok" if (changed and status in ("completed", "succeeded", "ok")) else (status or "no-change"),
            "notes": f"{tel.get('turns', '?')} turns"}


def run_codna(repo, issue, copydir):
    """Codna's REAL product — engine-behind: Algenta repo-intelligence localization + Monte-Carlo govern
    (only validated patches) + Telys reranking (`codna fix --memory on`). Local runs use Codna's owned
    fixed runtime via `codna doctor --start-stack` when no developer remote runtime override is configured. Set
    CODNA_BENCH_MODE=local for the bare warm-sidecar execution head instead (no engine)."""
    if os.environ.get("CODNA_BENCH_MODE") == "local":
        return run_codna_local(repo, issue, copydir)
    return _codna_fix(repo, issue, use_memory=True)


def run_codna_nomem(repo, issue, _copydir):
    return _codna_fix(repo, issue, use_memory=False)


def run_cursor(repo, issue, copydir):
    key = os.environ.get("CURSOR_API_KEY")
    if not _which(CURSOR):
        return {"status": "skipped", "notes": "cursor-agent not installed"}
    if not key:
        return {"status": "skipped", "notes": "CURSOR_API_KEY unset"}
    _git(copydir, "stash", "-u")
    cmd = [CURSOR, "-p", FIX_PROMPT.format(issue=issue), "--api-key", key, "--output-format", "json", "--force"]
    if os.environ.get("CURSOR_MODEL"):
        cmd += ["--model", os.environ["CURSOR_MODEL"]]
    rc, out, err, dt = _run(cmd, cwd=copydir, timeout=int(os.environ.get("CURSOR_TIMEOUT_S", "900")))
    # Usage rides the type=="result" event (camelCase keys); works for json (1 object) + stream-json (NDJSON).
    meta = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            meta = ev
    if not meta:  # single JSON object spanning lines
        mblob = re.search(r"\{.*\}", out, re.DOTALL)
        if mblob:
            try:
                meta = json.loads(mblob.group(0))
            except json.JSONDecodeError:
                pass
    u = meta.get("usage", {}) or {}
    in_t, out_t, cache_r = u.get("inputTokens"), u.get("outputTokens"), u.get("cacheReadTokens", 0)
    model = os.environ.get("CURSOR_MODEL", "composer-2.5-fast")
    cost, cnote = _token_cost(model, input_tokens=in_t, output_tokens=out_t, cached_tokens=cache_r)
    changed = [ln for ln in _git(copydir, "diff", "--name-only").splitlines()]
    return {"engine": "Cursor", "model": model,
            "localized": ", ".join(changed) or "—", "context_in": in_t,
            "out_tokens": out_t, "cache_read": cache_r,
            "total_tokens": (in_t + out_t) if (in_t is not None and out_t is not None) else None,
            "cost_usd": cost, "cost_note": cnote or "Composer (subscription — no per-token rate)",
            "time_s": round(dt),
            "status": "ok" if changed else (meta.get("subtype") or (f"exit {rc}" if rc else "no-change")),
            "notes": meta.get("result", "")[:60] if isinstance(meta.get("result"), str) else ""}


def run_codex(repo, issue, copydir):
    if not _which(CODEX):
        return {"status": "skipped", "notes": "codex not installed"}
    if not (os.path.exists(os.path.expanduser("~/.codex/auth.json")) or os.environ.get("OPENAI_API_KEY")):
        return {"status": "skipped", "notes": "codex not logged in + OPENAI_API_KEY unset"}
    _git(copydir, "stash", "-u")
    # --json → NDJSON events; usage rides turn.completed (snake_case). Parse STDOUT only (stderr has
    # non-JSON log noise); close stdin or codex hangs waiting for input; --skip-git-repo-check for safety.
    cmd = [CODEX, "exec", "--json", "--full-auto", "--skip-git-repo-check", "-C", copydir,
           FIX_PROMPT.format(issue=issue)]
    if os.environ.get("CODEX_MODEL"):
        cmd += ["--model", os.environ["CODEX_MODEL"]]
    rc, out, err, dt = _run(cmd, timeout=int(os.environ.get("CODEX_TIMEOUT_S", "900")), stdin=subprocess.DEVNULL)
    in_t = out_t = cached_t = None
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "turn.completed":
            u = ev.get("usage", {}) or {}
            in_t, out_t, cached_t = u.get("input_tokens"), u.get("output_tokens"), u.get("cached_input_tokens", 0)
    if in_t is None:  # fallback: human-mode "tokens used N" aggregate
        tok = re.search(r"tokens used\s+([\d,]+)", out + err)
        in_t = int(tok.group(1).replace(",", "")) if tok else None
    model = os.environ.get("CODEX_MODEL", "gpt-5.4")
    cost, cnote = _token_cost(model, input_tokens=in_t, output_tokens=out_t, cached_tokens=cached_t)
    changed = [ln for ln in _git(copydir, "diff", "--name-only").splitlines()]
    return {"engine": "Codex", "model": model,
            "localized": ", ".join(changed) or "—",
            "context_in": in_t, "out_tokens": out_t, "cache_read": cached_t,
            "total_tokens": (in_t + out_t) if (in_t is not None and out_t is not None) else None,
            "cost_usd": cost, "cost_note": cnote, "time_s": round(dt),
            "status": "ok" if changed else (f"exit {rc}" if rc else "no-change"), "notes": ""}


def run_cline(repo, issue, copydir):
    """Public Cline — the original, available-to-all CLI — with NO Algenta engine. The honest baseline
    for what the engine adds to Codna (Codna = Cline head + Algenta engine; this is the bare head). Runs
    headless (act mode + auto-approve); usage/cost ride the run_result JSON event. Skips (not crashes) if
    cline is missing or unauthenticated — auth via a Cline account or CLINE_PROVIDER/CLINE_API_KEY."""
    if not _which(CLINE):
        return {"status": "skipped", "notes": "cline not installed"}
    _git(copydir, "stash", "-u")
    hooks = os.environ.get("CLINE_HOOKS_DIR", "/tmp/cline-nohooks")  # empty dir bypasses the broken session.hook
    os.makedirs(hooks, exist_ok=True)
    # FRESH isolated data-dir per run: stale ~/.cline session/hub state otherwise dies with
    # "session not found" / "hook dispatch failed" on agentic (auto-approve) runs.
    datadir = tempfile.mkdtemp(prefix="cline-data-")
    cmd = [CLINE, "--json", "--auto-approve", "true", "--hooks-dir", hooks, "--data-dir", datadir,
           "-c", copydir, FIX_PROMPT.format(issue=issue)]
    if os.environ.get("CLINE_THINKING"):  # opt-in only — openai/gpt-5.4 rejects a 'thinking' param ("Unknown parameter")
        cmd += ["--thinking", os.environ["CLINE_THINKING"]]
    if os.environ.get("CLINE_PROVIDER"):
        cmd += ["-P", os.environ["CLINE_PROVIDER"]]
    if os.environ.get("CLINE_MODEL"):
        cmd += ["-m", os.environ["CLINE_MODEL"]]
    if os.environ.get("CLINE_API_KEY"):
        cmd += ["-k", os.environ["CLINE_API_KEY"]]
    rc, out, err, dt = _run(cmd, cwd=copydir, timeout=int(os.environ.get("CLINE_TIMEOUT_S", "900")))
    meta = {}
    for line in out.splitlines():                       # NDJSON; the run_result event carries usage+cost
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "run_result":
            meta = ev
    u = (meta.get("aggregateUsage") or meta.get("usage") or {})
    in_t, out_t, cache_r = u.get("inputTokens"), u.get("outputTokens"), u.get("cacheReadTokens", 0)
    model = ((meta.get("model") or {}).get("id")) or os.environ.get("CLINE_MODEL", "cline")
    cost = u.get("totalCost") or None
    if not cost:  # cline reports totalCost 0 for BYO providers → derive from tokens (same basis as codna/codex)
        cost, _ = _token_cost(model, input_tokens=in_t, output_tokens=out_t, cached_tokens=cache_r)
    finish, text = meta.get("finishReason"), (meta.get("text") or "")
    if finish == "error" and "nauthorized" in text:    # surface the auth gap as a skip, not a fake result
        return {"status": "skipped", "notes": "cline not authenticated (Cline account or CLINE_PROVIDER+CLINE_API_KEY)"}
    changed = [ln for ln in _git(copydir, "diff", "--name-only").splitlines()]
    return {"engine": "Cline", "model": model, "localized": ", ".join(changed) or "—",
            "context_in": in_t, "out_tokens": out_t, "cache_read": cache_r,
            "total_tokens": (in_t + out_t) if (in_t is not None and out_t is not None) else None,
            "cost_usd": round(cost, 4) if isinstance(cost, (int, float)) else None,
            "cost_note": "Cline-reported totalCost", "time_s": round(dt),
            "status": "ok" if changed else (finish or (f"exit {rc}" if rc else "no-change")),
            "notes": text[:60]}


RUNNERS = {"codna": run_codna, "codna-nomem": run_codna_nomem,
           "cursor": run_cursor, "codex": run_codex, "cline": run_cline}

# ── security mode ───────────────────────────────────────────────────────────────────────────────
# Codna's security headline is REACHABILITY PROOF: ingest a scanner SARIF and prove (0 LLM tokens)
# which findings are production-reachable — a capability Cursor/Codex don't have. The comparable axis
# is the agentic FIX of the vulns; the differentiator is the proof + the closure barrier check.
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VULN_FIXTURE = os.path.join(HERE, "fixtures", "vulnapp")
SEC_PROMPT = ("This Python web app has user-input-to-sink security vulnerabilities (SQL injection, "
              "command injection, and path traversal — user request data flowing into a SQL query, a "
              "shell command, and a file path). Fix ALL of them with minimal, safe changes "
              "(parameterized queries, no shell / shlex.quote, path validation). Do not ask questions.")
# Recognized remediation barriers (mirrors the reference engine's closure check) — used to score fixes.
_SANITIZER = re.compile(
    r"parameteri|saniti[sz]e|escape|shlex\.quote|bleach|html\.escape|secure_filename|abspath|realpath|"
    r"prepared\s*statement|placeholder|bindparam|shell\s*=\s*False|execute\([^)]*,\s*[\(\[]", re.I)
_VULN_FINDINGS = [  # (rule, cwe, severity, file, sink-pattern, expected)
    ("py/sql-injection", "cwe-089", "9.8", "app.py", "SELECT * FROM orders", "production-reachable"),
    ("py/command-line-injection", "cwe-078", "9.8", "app.py", "subprocess.check_output", "production-reachable"),
    ("py/path-injection", "cwe-022", "7.5", "app.py", "send_file(os.path.join", "production-reachable"),
    ("py/sql-injection", "cwe-089", "9.8", "legacy.py", "SELECT * FROM legacy", "not-production-reachable"),
]
_VULN_MSG = {"py/sql-injection": "User input flows into a SQL query.",
             "py/command-line-injection": "User input flows into a shell command.",
             "py/path-injection": "User input flows into a file path."}


def _materialize_vuln_repo(workdir):
    """Copy the bundled vulnerable fixture into a fresh git repo (materializing .py.txt -> .py) and
    emit a CodeQL-shaped SARIF bound to the commit. Returns (repo_path, sarif_path)."""
    repo = os.path.join(workdir, "vulnapp")
    shutil.copytree(DEFAULT_VULN_FIXTURE, repo)
    for fn in os.listdir(repo):
        if fn.endswith(".py.txt"):
            os.rename(os.path.join(repo, fn), os.path.join(repo, fn[:-4]))
    env = {**os.environ, "GIT_AUTHOR_NAME": "b", "GIT_AUTHOR_EMAIL": "b@b.co",
           "GIT_COMMITTER_NAME": "b", "GIT_COMMITTER_EMAIL": "b@b.co"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "vulnerable fixture"], cwd=repo, check=True, env=env)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    def _line(path, needle):
        for i, ln in enumerate(open(os.path.join(repo, path), encoding="utf-8"), 1):
            if needle in ln:
                return i
        return 1
    rules, results, seen = [], [], set()
    for rule, cwe, sev, fname, needle, _ in _VULN_FINDINGS:
        if rule not in seen:
            rules.append({"id": rule, "properties": {"security-severity": sev, "tags": ["security", f"external/cwe/{cwe}"]}})
            seen.add(rule)
        results.append({"ruleId": rule, "message": {"text": _VULN_MSG[rule]},
                        "locations": [{"physicalLocation": {"artifactLocation": {"uri": fname},
                                      "region": {"startLine": _line(fname, needle)}}}]})
    sarif = {"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
             "runs": [{"tool": {"driver": {"name": "CodeQL", "version": "2.15.0", "rules": rules}},
                       "versionControlProvenance": [{"revisionId": sha, "repositoryUri": "https://github.com/acme/vulnapp"}],
                       "results": results}]}
    sarif_path = os.path.join(workdir, "scan.sarif")
    json.dump(sarif, open(sarif_path, "w"))
    return repo, sarif_path


def sec_codna(repo, sarif, _copy):
    """Codna's reachability triage (0 LLM) — the security moat Cline/Cursor/Codex lack."""
    if not os.environ.get("CODNA_API_KEY"):
        return {"engine": "Codna (secure)", "status": "skipped", "notes": "CODNA_API_KEY unset"}
    rc, out, err, dt = _run([CODNA, "secure", repo, "--from-sarif", sarif, "--engine", "remote", "--json"], timeout=600)
    m = re.search(r"\{.*\}", out, re.DOTALL)
    d = json.loads(m.group(0)) if m else {}
    f = d.get("findings", [])
    counts = d.get("counts", {})
    reach = counts.get("production-reachable", sum(1 for x in f if x.get("classification") == "production-reachable"))
    return {"engine": "Codna (secure)", "capability": "reachability proof (0 LLM)",
            "result": f"{reach}/{len(f)} production-reachable" + (f", {counts.get('unreachable',0)} unreachable" if counts.get("unreachable") else ""),
            "time_s": round(dt), "status": "ok" if f else (f"exit {rc}" if rc else "no-findings"),
            "notes": "discriminates reachable vs dead code; cuts scanner noise pre-fix"}


def _sec_agentic(engine_label, cmd_fn, repo, copydir, model):
    _git(copydir, "stash", "-u")
    rc, out, err, dt, blob = cmd_fn(copydir)
    diff = _git(copydir, "diff")
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    barrier_hits = sum(1 for line in added if _SANITIZER.search(line))
    files = [ln for ln in _git(copydir, "diff", "--name-only").splitlines()]
    return {"engine": engine_label, "capability": "agentic fix (no reachability signal)",
            "result": f"{len(files)} file(s), {barrier_hits} sanitizer barrier(s) added" if files else "no change",
            "model": model, "time_s": round(dt), "status": "ok" if files else (f"exit {rc}" if rc else "no-change"),
            "notes": "fixes blindly — no proof of which findings are reachable"}


def sec_cursor(repo, _sarif, copydir):
    key = os.environ.get("CURSOR_API_KEY")
    if not _which(CURSOR) or not key:
        return {"engine": "Cursor", "status": "skipped", "notes": "cursor-agent/CURSOR_API_KEY missing"}
    def _cmd(cd):
        c = [CURSOR, "-p", SEC_PROMPT, "--api-key", key, "--output-format", "json", "--force"]
        rc, o, e, dt = _run(c, cwd=cd, timeout=int(os.environ.get("CURSOR_TIMEOUT_S", "900")))
        return rc, o, e, dt, o + e
    return _sec_agentic("Cursor", _cmd, repo, copydir, os.environ.get("CURSOR_MODEL", "default"))


def sec_cline(repo, _sarif, copydir):
    if not _which(CLINE):
        return {"engine": "Cline", "status": "skipped", "notes": "cline not installed"}
    hooks = os.environ.get("CLINE_HOOKS_DIR", "/tmp/cline-nohooks")
    os.makedirs(hooks, exist_ok=True)
    datadir = tempfile.mkdtemp(prefix="cline-data-")
    model = os.environ.get("CLINE_MODEL", "cline")

    def _cmd(cd):
        c = [CLINE, "--json", "--auto-approve", "true", "--hooks-dir", hooks, "--data-dir", datadir,
             "-c", cd, SEC_PROMPT]
        if os.environ.get("CLINE_THINKING"):
            c += ["--thinking", os.environ["CLINE_THINKING"]]
        if os.environ.get("CLINE_PROVIDER"):
            c += ["-P", os.environ["CLINE_PROVIDER"]]
        if os.environ.get("CLINE_MODEL"):
            c += ["-m", os.environ["CLINE_MODEL"]]
        if os.environ.get("CLINE_API_KEY"):
            c += ["-k", os.environ["CLINE_API_KEY"]]
        rc, o, e, dt = _run(c, cwd=cd, timeout=int(os.environ.get("CLINE_TIMEOUT_S", "900")))
        return rc, o, e, dt, o + e

    return _sec_agentic("Cline", _cmd, repo, copydir, model)


def sec_codex(repo, _sarif, copydir):
    if not _which(CODEX) or not (os.path.exists(os.path.expanduser("~/.codex/auth.json")) or os.environ.get("OPENAI_API_KEY")):
        return {"engine": "Codex", "status": "skipped", "notes": "codex not logged in"}
    def _cmd(cd):
        rc, o, e, dt = _run([CODEX, "exec", SEC_PROMPT, "--full-auto", "-C", cd],
                            timeout=int(os.environ.get("CODEX_TIMEOUT_S", "900")))
        return rc, o, e, dt, o + e
    return _sec_agentic("Codex", _cmd, repo, copydir, os.environ.get("CODEX_MODEL", "gpt-5.4"))


SEC_RUNNERS = {"codna": sec_codna, "cline": sec_cline, "cursor": sec_cursor, "codex": sec_codex}


def _resolve_repo(repo, ref, workdir):
    """Clone a URL (shallow) into a unique dir under workdir, or use a local path. Returns the base."""
    if repo.startswith(("http://", "https://", "git@")) or repo.endswith(".git"):
        # Unique per-repo dir (basename), so multiple repos in one --random run don't collide.
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", repo.rstrip("/").split("/")[-1].removesuffix(".git")) or "repo"
        dst = tempfile.mkdtemp(prefix=f"clone-{name}-", dir=workdir)
        shutil.rmtree(dst)
        cmd = ["git", "clone", "--depth", "1"] + (["--branch", ref] if ref else []) + [repo, dst]
        rc, _, err, _ = _run(cmd, timeout=300)
        if rc != 0:
            print(f"clone failed for {repo}: {err.strip()[:120]} — skipping", file=sys.stderr)
            return None
        return dst
    return os.path.abspath(os.path.expanduser(repo))


def _copy_repo(base: str, workdir: str, prefix: str) -> str:
    dst = tempfile.mkdtemp(prefix=prefix, dir=workdir)
    shutil.rmtree(dst)
    shutil.copytree(base, dst, symlinks=True)
    return dst


def _bench_one(repo, issue, engines, ref, workdir, precheck_cmd=None, verify_cmd=None):
    """Run the selected engines on one (repo, issue) — IN PARALLEL (independent processes), so the
    per-repo wall-clock is the slowest single engine, not the sum. Returns (repo_name, rows)."""
    base = _resolve_repo(repo, ref, workdir)
    if base is None:  # clone failed — skip this repo, don't abort the run
        name = repo.rstrip("/").split("/")[-1].removesuffix(".git") or "repo"
        return name, [{"engine": e, "status": "skipped", "notes": "clone failed"} for e in engines]
    repo_name = os.path.basename(base.rstrip("/")) or "repo"
    precheck_dir = _copy_repo(base, workdir, f"precheck-{repo_name}-")
    precheck = _run_precheck(precheck_dir, precheck_cmd)
    if not precheck.get("precheck_ok"):
        rows = []
        for eng in engines:
            row = {"engine": eng, "status": "skipped", "notes": precheck.get("precheck_notes", "precheck failed")}
            _attach_precheck(row, precheck)
            _verify_unavailable(row, "precheck did not establish a failing baseline")
            rows.append(row)
        return repo_name, rows
    # Pre-create isolated checkouts sequentially (copytree is not safe to interleave); codna reads
    # `base` in place (its snapshot is server-side). Each cursor/codex run gets its own dir.
    targets = {}
    for i, eng in enumerate(engines):
        # every engine (codna's local agent included) edits its OWN isolated checkout in place
        targets[eng] = _copy_repo(base, workdir, f"{eng}-{repo_name}-{i}-")

    def _go(eng):
        print(f"[{repo_name}/{eng}] running …", file=sys.stderr)
        r = RUNNERS[eng](targets[eng], issue, targets[eng])
        r.setdefault("engine", eng)
        _attach_precheck(r, precheck)
        _attach_verification(r, eng, targets[eng], verify_cmd)
        print(f"[{repo_name}/{eng}] {r.get('status')} — {r.get('localized', '—')} ({r.get('time_s', '?')}s)",
              file=sys.stderr)
        return r

    # ThreadPool: each runner blocks on its own subprocess (GIL released), so they run concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as ex:
        rows = list(ex.map(_go, engines))  # map preserves engine order
    return repo_name, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Codna vs Cursor vs Cline — side-by-side fix benchmark on public repos.",
        epilog="Examples:\n"
               "  benchmark_suite.py --repo https://github.com/pallets/flask --issue 'send_file webp type'\n"
               "  benchmark_suite.py --random 3 --engines codna-nomem,codna,cline,cursor --out report.md\n"
               "Engines auto-skip (with a noted reason) if their CLI/creds are absent — so external\n"
               "users can run any subset. See cli/bench/README.md for per-engine setup.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="public GitHub URL or local path")
    ap.add_argument("--issue", help="the bug/issue to fix (required with --repo)")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="benchmark N random curated public repos (each with a realistic issue)")
    ap.add_argument("--engines", default=DEFAULT_ENGINES,
                    help="comma list: codna-nomem,codna,cline,cursor,codex")
    ap.add_argument("--mode", choices=["fix", "security", "all"], default="all",
                    help="all (default: fix + security) | fix (agentic bug fix) | security (Codna "
                         "reachability proof + selected-agent vuln fix)")
    ap.add_argument("--sarif", help="security mode: scanner SARIF (default: the bundled vulnapp fixture)")
    ap.add_argument("--ref", help="branch/tag to clone (single --repo only)")
    ap.add_argument("--seed", type=int, help="seed for --random (reproducible selection)")
    ap.add_argument("--precheck-cmd",
                    help="optional command that must fail on the baseline before engines run")
    ap.add_argument("--verify-cmd",
                    help="optional deterministic command to run after each local-patch engine, e.g. 'pytest -q'")
    ap.add_argument("--out", help="markdown report path (default: stdout)")
    args = ap.parse_args(argv)

    try:
        engines = _parse_engines(args.engines)
        precheck_cmd = _parse_precheck_cmd(args.precheck_cmd)
        verify_cmd = _parse_verify_cmd(args.verify_cmd)
    except ValueError as exc:
        sys.exit(str(exc))

    # fix jobs (only if the user gave a target; security runs on the bundled fixture regardless)
    jobs = []
    if args.mode in ("fix", "all"):
        if args.random:
            try:
                jobs = _select_random_jobs(args.random, args.seed)
            except ValueError as exc:
                sys.exit(str(exc))
        elif args.repo and args.issue:
            jobs = [(args.repo, args.issue)]
        elif args.mode == "fix":
            sys.exit("fix mode: provide --repo URL --issue '…', or --random N")

    workdir = tempfile.mkdtemp(prefix="codna-suite-")
    sections = []  # ("fix"|"security", str)

    def _emit():  # write the combined report so far (incremental crash-safety)
        report = "\n".join(s for _, s in sections)
        if args.out:
            with open(args.out, "w") as f:
                f.write(report)
        return report

    fix_results = []
    for repo, issue in jobs:
        name, rows = _bench_one(repo, issue, engines, args.ref, workdir, precheck_cmd, verify_cmd)
        fix_results.append((name, repo, issue, rows))
        sections = [("fix", render_report(fix_results))]
        _emit()  # persist after each repo

    if args.mode in ("security", "all"):
        sections.append(("security", _security_section(args, engines, workdir)))

    if not sections:
        sys.exit("nothing to run — provide --repo/--random for fix, and/or use --mode security/all")
    report = _emit()
    if args.out:
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


def render_report(results) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = ["# Codna vs Cursor vs Cline — fix benchmark\n",
           f"_Generated {now} · {len(results)} repo(s) · each engine runs the same issue headless on its "
           f"own checkout, no shared infra._\n"]
    for name, url, issue, rows in results:
        out.append(render(name, url, issue, rows))
    out.append("## Notes\n- Each engine runs the SAME issue headless on its own checkout — no shared infra.\n"
               "- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only "
               "difference is whether Telys code memory recalls `related_symbols` into the engine signals. "
               "Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model "
               "auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized "
               "(one local engine) so the timing delta is clean.\n"
               "- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and "
               "reports planner usage as split input/output/cache/total tokens; Cursor/Cline figures are their own "
               "agents' consumption. Codna is inspect-mode (patch ref, no apply); Cline/Cursor/Codex edit a throwaway copy.\n"
               "- **Precheck** is `fail-first` only when `--precheck-cmd` fails on a clean baseline. If it "
               "passes or times out, engines are skipped for that repo to avoid false fix-rate claims.\n"
               "- **Final verify** is `pass`/`fail` when `--verify-cmd` runs after a local patch. "
               "**Fix verified** is stricter: `verified` only means precheck failed first and final verify passed.\n"
               "- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.\n"
               "- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Cline/Codex CLIs report detailed "
               "**tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact "
               "reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is "
               "subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.\n")
    return "\n".join(out)


def render(repo_name, repo_url, issue, rows) -> str:
    o = [f"## {repo_name}\n",
         f"_repo: {repo_url} · issue: {issue}_\n",
         "| Engine | Model | Localized | In tok | Out tok | Cache tok | Total tok | Evidence | Memory (Telys) | Time | Cost | Precheck | Final verify | Fix verified | Status |",
         "|---|---|---|--:|--:|--:|--:|---|---|--:|--:|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "skipped":
            o.append(f"| {r.get('engine')} | — | — | — | — | — | — | — | — | — | — | "
                     f"{r.get('precheck','n/a')} | {r.get('verification','n/a')} | {r.get('fix_result','n/a')} | "
                     f"skipped: {r.get('notes','')} |")
            continue
        evidence = r.get("context_note") or (f"{r['evidence_tokens']:,}" if r.get("evidence_tokens") else "—")
        if r.get("cost_usd") is not None:
            real = r.get("engine", "").startswith("Codna")
            cost = f"${r['cost_usd']:.3f}" + ("" if real else " est")  # Codna = real; others derived from tokens
        else:
            cost = r.get("cost_note") or "—"
        o.append(f"| {r.get('engine')} | {r.get('model','—')} | {r.get('localized','—')[:60]} | "
                 f"{_fmt_tokens(r.get('context_in'))} | {_fmt_tokens(r.get('out_tokens'))} | "
                 f"{_fmt_tokens(r.get('cache_read'))} | {_fmt_tokens(r.get('total_tokens'))} | "
                 f"{evidence} | {r.get('memory','—')} | {r.get('time_s','—')}s | {cost} | "
                 f"{r.get('precheck','n/a')} | {r.get('verification','n/a')} | {r.get('fix_result','n/a')} | "
                 f"{r.get('status','—')} |")
    # ── A/B delta: Codna +Telys vs −Telys (what the memory changed, same issue/engine/conditions) ──
    by = {r.get("engine"): r for r in rows}
    on, off = by.get("Codna +Telys"), by.get("Codna −Telys")
    if on and off and on.get("status") == "ok" and off.get("status") == "ok":
        dt_ = (on.get("time_s") or 0) - (off.get("time_s") or 0)
        ca, cb = on.get("cost_usd"), off.get("cost_usd")
        dc = (ca - cb) if (ca is not None and cb is not None) else None
        same_loc = on.get("localized") == off.get("localized")
        nrec = len(on.get("related_symbols") or [])
        o.append("")
        o.append(f"_**Memory A/B** (same issue, engine-behind): Telys recalled **{nrec}** related symbol(s); "
                 f"localization **{'unchanged' if same_loc else 'CHANGED'}** vs −Telys; "
                 f"Δtime {dt_:+}s" + (f" · Δcost ${dc:+.3f}" if dc is not None else "") + "._")
    o.append("")
    return "\n".join(o)


def _fmt_tokens(value) -> str:
    if isinstance(value, bool):
        return "—"
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return "—"


def _security_section(args, engines, workdir):
    """Security: Codna proves reachability (0 LLM); all selected engines attempt the vuln fix.
    Returns the report section string (caller writes it)."""
    # Security is reachability proof — memory-independent. Collapse the codna-nomem A/B variant into the
    # single Codna security row (no duplicate reachability run).
    engines = [e for e in engines if e != "codna-nomem"]
    secdir = os.path.join(workdir, "security")
    os.makedirs(secdir, exist_ok=True)
    if args.repo and args.sarif:
        repo = _resolve_repo(args.repo, args.ref, secdir)
        if repo is None:
            return "## Security\n\n_skipped: clone failed_\n"
        sarif, label = os.path.abspath(os.path.expanduser(args.sarif)), args.repo
    else:
        repo, sarif = _materialize_vuln_repo(secdir)
        label = "bundled vulnapp fixture (3 live-route vulns + 1 dead-code)"
    repo_name = os.path.basename(repo.rstrip("/")) or "repo"
    targets = {}
    for i, eng in enumerate(engines):
        targets[eng] = repo if eng == "codna" else os.path.join(secdir, f"{eng}-{repo_name}-{i}")
        if eng != "codna":
            shutil.copytree(repo, targets[eng], symlinks=True)

    def _go(eng):
        print(f"[security/{eng}] running …", file=sys.stderr)
        r = SEC_RUNNERS[eng](repo, sarif, targets[eng])
        r.setdefault("engine", eng)
        print(f"[security/{eng}] {r.get('status')} — {r.get('result', '—')} ({r.get('time_s', '?')}s)",
              file=sys.stderr)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as ex:
        rows = list(ex.map(_go, engines))
    return security_render(repo_name, label, rows)


def security_render(repo_name, label, rows) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    o = [f"# Security benchmark — {repo_name}\n",
         f"_Generated {now} · target: {label} · Codna proves reachability (0 LLM); selected agents attempt the fix._\n",
         "| Engine | Capability | Result | Time | Status |",
         "|---|---|---|--:|---|"]
    for r in rows:
        if r.get("status") == "skipped":
            o.append(f"| {r.get('engine')} | — | — | — | skipped: {r.get('notes','')} |")
            continue
        o.append(f"| {r.get('engine')} | {r.get('capability','—')} | {r.get('result','—')} | "
                 f"{r.get('time_s','—')}s | {r.get('status','—')} |")
    o.append("\n## Notes\n- **Codna** ingests the scanner SARIF and PROVES which findings are "
             "production-reachable (0 LLM tokens) — telling real risk from dead code — which Cline/Cursor/Codex "
             "cannot. On the bundled fixture (3 live-route vulns + 1 dead-code helper) Codna should report "
             "3 production-reachable, 1 unreachable.\n- Cline/Cursor/Codex fix agentically with **no reachability "
             "signal** (they patch whatever they find); 'sanitizer barriers' counts recognized remediations "
             "(parameterized query / shlex.quote / path validation) in their diff.\n- `exploitable` "
             "(independent taint proof) is Codna's wired-but-off-in-v1 seam; closure is a barrier check, not "
             "taint re-verification — stated honestly, not overclaimed.\n")
    return "\n".join(o)


if __name__ == "__main__":
    raise SystemExit(main())
