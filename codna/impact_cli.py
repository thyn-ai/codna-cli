"""Codna repo intelligence on the CLI: test-impact analysis + code-memory export.

``codna impact``         which tests a diff can affect (codna.impact engine — multi-language
                         classification, precise Python import-graph edges, run-MORE-never-fewer).
``codna memory export``  the repo's code memory as an ultra-small READ-ONLY serve artifact
                         (telys compact tier: quantized slab + provenance, ~5x smaller).

Both are deterministic and offline: no engine, no LLM tokens.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def _git_changed(repo_root: str, base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", repo_root, "diff", "--name-only", f"{base}...{head}"],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"git diff {base}...{head} failed: {out.stderr.strip()}")
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def cmd_impact(args) -> int:
    """Repo-aware test selection: print the tests a diff can affect (``ALL`` = run the full suite).

    Output contract (kept stable for CI consumers): one test path per line on stdout, the reason on
    stderr; ``ALL`` when the full suite must run; ``--json`` emits {mode, tests, reason}."""
    from . import codeunits
    from .impact import compute_impact

    repo = os.path.abspath(os.path.expanduser(args.repo or "."))
    changed = list(args.changed) if args.changed is not None else _git_changed(repo, args.base, args.head)
    res = compute_impact(repo, changed, codeunits=codeunits,
                         shared_core_fraction=args.shared_core_fraction)
    if args.as_json:
        print(json.dumps({"mode": res.mode, "tests": res.tests, "reason": res.reason}))
    elif res.mode == "all":
        print("ALL")
        print(f"# reason: {res.reason}", file=sys.stderr)
    else:
        for t in res.tests:
            print(t)
        print(f"# {res.reason}", file=sys.stderr)
    return 0


def cmd_memory(args) -> int:
    """``codna memory export``: write the repo's code memory as a compact serve artifact.

    Indexes first when the memory is empty, refreshes when stale (both inside
    CodeMemory.export_serve_artifact), then seals the artifact. The artifact is self-contained and
    read-only — open it with telys' ``open_compact``."""
    from .memory import CodeMemory, CodeMemoryError

    repo = os.path.abspath(os.path.expanduser(args.repo or "."))
    try:
        info = CodeMemory(repo).export_serve_artifact(args.path, mode=args.mode)
    except CodeMemoryError as exc:
        print(json.dumps({"error": {"code": "memory_export", "message": str(exc)}}), file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2))
    return 0


def register_cli(sub) -> None:
    """Register `codna impact` + `codna memory` on the CLI subparser (kept out of cli.py's 1000-line
    ceiling; same split pattern as byok_cli/mcp_cli/webhook_cli)."""
    pim = sub.add_parser("impact", help="Which tests a diff can affect (repo-aware test selection).")
    pim.add_argument("--repo", help="repo root (default: current directory)")
    pim.add_argument("--base", default="origin/main", help="diff base (default origin/main)")
    pim.add_argument("--head", default="HEAD", help="diff head (default HEAD)")
    pim.add_argument("--changed", nargs="*", default=None, help="explicit changed files (skips git)")
    pim.add_argument("--shared-core-fraction", dest="shared_core_fraction", type=float, default=0.34,
                     help="affected-share above which the full suite runs (default 0.34)")
    pim.add_argument("--json", dest="as_json", action="store_true", help="machine-readable output")
    pim.set_defaults(func=cmd_impact)

    pmem = sub.add_parser("memory", help="Codna code memory: the on-device repo index (Telys).")
    pmem_sub = pmem.add_subparsers(dest="memory_action")
    pme = pmem_sub.add_parser(
        "export", help="Export the index as an ultra-small read-only serve artifact (~4x smaller).")
    pme.add_argument("path", help="artifact directory to write")
    pme.add_argument("--mode", default="int8", choices=("int8", "pq"),
                     help="quantization tier: int8 (default, ~4x) or pq (super-large repos)")
    pme.add_argument("--repo", help="repo root (default: current directory)")
    pme.set_defaults(func=cmd_memory)
