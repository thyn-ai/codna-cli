"""`codna run verify-ledger` — the auditor's command, and it PRINTS divergence instead of crashing.

Split out of `run_cli.py` for the repository's 1000-line ceiling (`test_modularity.py`), on the seam
this package already uses: a command that only READS is not part of starting or executing a run.

Two requirements make this more than a wrapper around `ledger.verify_chain`:

1. **Print, never crash.** A tampered ledger is exactly the input this command exists for, and
   `verify_chain` raises `LedgerError` on a malformed line. So the walk is `verify_ledger.verify`,
   which turns every failure into a located finding and still exits with a code an operator can act
   on. The JSON body is ALWAYS printed, before the exit code matters.
2. **A chain must not be beatable by shortening it.** ``--expect-last-seq`` lets a caller assert the
   head they remember (a client's resume point, `last_seq` from a SIEM row); a chain that verifies
   internally but is SHORTER than that is reported as `shrank`, which is the only way to catch a
   removed tail.

Exit codes: 0 intact · 2 diverged (a verdict) · 1 the command could not run (no ledger).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["add_verify_ledger_flags", "cmd_verify_ledger"]


def cmd_verify_ledger(args: Any) -> int:
    """The command body. Prints one JSON report and returns the verdict's exit code."""
    from .verify_ledger import EXIT_DIVERGED, EXIT_INTACT, verify_ledger_file

    # `--run-id` is the spelling under `codna run` (where `--run` would abbreviate ambiguously against
    # `--run-id`/`--run-timeout`); `--run` is the spelling under `codna durable`. One handler, both
    # flags, so the two surfaces cannot drift.
    run_id = str(getattr(args, "run", None) or getattr(args, "run_id", "") or "").strip()
    if not run_id:
        _fail("verify-ledger needs a run id.", code="durable_run_required")
        return 1
    expected = getattr(args, "expect_last_seq", None)
    if expected is not None and (isinstance(expected, bool) or expected < 0):
        _fail("--expect-last-seq must be a non-negative integer.", code="durable_seq_invalid")
        return 1

    path = _ledger_path(args, run_id)
    report = verify_ledger_file(path, expected_records=expected)
    body = report.to_payload()
    body["run_id"] = run_id
    # `problems` mirrors the long-standing `codna durable ledger` field so an existing reader of that
    # body keeps working, and gains the located hashes for free.
    body["problems"] = [finding.describe() for finding in report.findings]
    _print(body)
    if not report.exists:
        # Not a verdict about the chain: this host has nothing to verify. EXIT_FAILURE.
        return 1
    return EXIT_INTACT if report.intact else EXIT_DIVERGED


def _ledger_path(args: Any, run_id: str) -> Path:
    """Resolve the run's ledger path, including for a run this host has no config for.

    `config` and `run_cli` are imported INSIDE the function on purpose: `run_cli` imports this module
    for its registration, so a module-scope import here closes a cycle that only appears when
    `verify_ledger_cli` is the first of the three to be imported (measured: `python -c "import
    codna.durable.verify_ledger_cli"` → ImportError naming `DEFAULT_RUN_TIMEOUT_S`).
    """
    from .config import load_durable_config, validate_run_id

    validate_run_id(run_id)
    # Resolve the config the way every durable command does, without importing the CLI's own
    # `_config` (which would reintroduce the cycle). The two knobs a ledger path depends on are the
    # runtime root and an explicit runs override, both of which `load_durable_config` reads itself.
    import os

    config = load_durable_config(
        env=dict(os.environ), profile=getattr(args, "temporal_profile", None)
    )
    return config.ledger_path(run_id)


def _print(payload: dict[str, Any]) -> None:
    # `cli` is imported at call time: `run_cli` imports THIS module and `cli` imports `run_cli`, so a
    # module-scope import here reintroduces the cycle that broke a cold `verify_ledger_cli` import.
    from .cli import _print as print_json

    print_json(payload)


def _fail(message: str, *, code: str) -> None:
    """Report a usage problem as the repository's structured error, never a traceback."""
    from .cli import _print as print_json

    print_json({"error": {"code": code, "message": message, "details": {}}})


def add_verify_ledger_flags(verify: Any, *, flag: str = "--run") -> None:
    """Flags for both surfaces (`codna run verify-ledger`, `codna durable verify-ledger`).

    ``flag`` differs because under `codna run` the option string ``--run`` would ABBREVIATE
    ambiguously against the parent's ``--run-id``/``--run-timeout`` (argparse refuses it with
    "ambiguous option"). The command is the same; only its spelling on that surface differs.
    """
    verify.add_argument("--run-id" if flag == "--run-id" else flag, required=True, dest="run",
                        help="run id whose ledger to verify")
    verify.add_argument(
        "--expect-last-seq",
        type=int,
        default=None,
        dest="expect_last_seq",
        help="the last seq you remember; a chain shorter than this is reported as truncated",
    )
