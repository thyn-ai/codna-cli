"""`codna durable audit` — the operational half of the SIEM export and retention story.

The classes in `siem.py` are the mechanism; this is what an operator actually runs. Two operations,
both idempotent, both reporting what they did rather than what they hoped:

* ``--apply-retention`` prunes expired SIEM export files and reports the filenames it removed. It
  never deletes a ledger (see `siem.RetentionPolicy`), because the ledger is the evidence the
  retention window exists to preserve.
* ``--verify-ledgers`` walks every run ledger this host still holds and reports the ones whose hash
  chain is broken. This is the answer to "the run's history expired from Temporal Cloud — is the
  copy we kept still intact?", which is the question that makes retention worth anything.

``--retention-days`` defaults from ``CODNA_SIEM_RETENTION_DAYS`` (minimum 90, because below Temporal
Cloud's own window there is nothing to outlive). Neither operation needs Temporal, the durable extra,
or a credential: they only read and unlink files under this host's runtime root.

Exit codes: 0 = everything asked for succeeded and no ledger is broken · 2 = a ledger failed
verification (a verdict) · 1 = the command could not run (a malformed setting).
"""
from __future__ import annotations

import os
from typing import Any

__all__ = ["add_audit_flags", "cmd_audit"]


def cmd_audit(args: Any) -> int:
    from .config import load_durable_config, validate_run_id  # noqa: F401 - validate: parity of errors
    from .siem import (
        RetentionPolicy,
        SiemDeliveryError,
        build_siem_sink,
        retention_days_from_env,
    )

    try:
        days = (
            int(args.retention_days)
            if getattr(args, "retention_days", None) is not None
            else retention_days_from_env()
        )
    except (TypeError, ValueError):
        _fail("--retention-days must be a whole number of days.", code="siem_retention_invalid")
        return 1
    except SiemDeliveryError as exc:
        _fail(str(exc), code=exc.code, details=dict(exc.details))
        return 1

    try:
        policy = RetentionPolicy(days=days)
    except ValueError as exc:
        _fail(str(exc), code="siem_retention_too_short", details={"days": days})
        return 1

    config = load_durable_config(
        env=dict(os.environ), profile=getattr(args, "temporal_profile", None)
    )
    body: dict[str, Any] = {
        "retention_days": policy.days,
        "siem_dir": str(policy.siem_dir),
        "runs_root": str(config.runs_root),
    }
    if getattr(args, "apply_retention", False):
        body["pruned"] = policy.apply()
    if getattr(args, "verify_ledgers", False):
        body["broken_ledgers"] = policy.verify_ledgers(config.runs_root)
    # The export's own location, so "where do I point my collector" is answerable without reading
    # code. A REMOTE sink is reported as unwired rather than left to be discovered in a runbook.
    body["export"] = {"sink": "file", "path": str(build_siem_sink().path)}
    body["export"]["remote"] = "not implemented in this build; see siem.RemoteSiemSink"
    _print(body)
    return 2 if body.get("broken_ledgers") else 0


def add_audit_flags(audit: Any) -> None:
    audit.add_argument(
        "--retention-days",
        type=int,
        default=None,
        dest="retention_days",
        help="the SIEM retention window (default: CODNA_SIEM_RETENTION_DAYS, else 365; min 90)",
    )
    audit.add_argument(
        "--apply-retention",
        action="store_true",
        dest="apply_retention",
        help="prune expired SIEM export files (never a ledger)",
    )
    audit.add_argument(
        "--verify-ledgers",
        action="store_true",
        dest="verify_ledgers",
        help="verify the hash chain of every run ledger still on this host",
    )


def _print(payload: dict[str, Any]) -> None:
    from .cli import _print as print_json

    print_json(payload)


def _fail(message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
    from .cli import _print as print_json

    print_json({"error": {"code": code, "message": message, "details": dict(details or {})}})
