from __future__ import annotations

import shlex
from collections.abc import Callable

RunResult = tuple[int, str, str, float]
RunFn = Callable[..., RunResult]


def parse_command_arg(flag: str, raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    try:
        cmd = shlex.split(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {flag}: {exc}") from exc
    if not cmd:
        raise ValueError(f"invalid {flag}: command cannot be empty")
    return cmd


def parse_precheck_cmd(raw: str | None) -> list[str] | None:
    return parse_command_arg("--precheck-cmd", raw)


def parse_verify_cmd(raw: str | None) -> list[str] | None:
    return parse_command_arg("--verify-cmd", raw)


def last_output_line(out: str, err: str, fallback: str) -> str:
    lines = (err or out).strip().splitlines()
    return lines[-1][:80] if lines else fallback


def run_precheck(repo: str, precheck_cmd: list[str] | None, *, run: RunFn, timeout: int) -> dict:
    if not precheck_cmd:
        return {"precheck": "n/a", "precheck_ok": True, "precheck_notes": "no --precheck-cmd configured"}
    rc, out, err, dt = run(precheck_cmd, cwd=repo, timeout=timeout)
    if rc == 124:
        return {
            "precheck": "timeout",
            "precheck_ok": False,
            "precheck_time_s": round(dt, 1),
            "precheck_notes": "precheck timed out",
        }
    if rc == 0:
        return {
            "precheck": "passed",
            "precheck_ok": False,
            "precheck_time_s": round(dt, 1),
            "precheck_notes": "precheck passed; issue was not reproduced",
        }
    return {
        "precheck": "fail-first",
        "precheck_ok": True,
        "precheck_rc": rc,
        "precheck_time_s": round(dt, 1),
        "precheck_notes": last_output_line(out, err, "precheck failed as expected"),
    }


def attach_precheck(row: dict, precheck: dict) -> dict:
    row["precheck"] = precheck.get("precheck", "n/a")
    row["precheck_notes"] = precheck.get("precheck_notes", "")
    if precheck.get("precheck_time_s") is not None:
        row["precheck_time_s"] = precheck["precheck_time_s"]
    return row


def verify_unavailable(row: dict, reason: str) -> dict:
    row["accuracy_available"] = False
    row["verified"] = None
    row["verification"] = "n/a"
    row["verification_notes"] = reason
    row["fix_result"] = "n/a"
    return row


def attach_verification(
    row: dict,
    engine: str,
    copydir: str,
    verify_cmd: list[str] | None,
    *,
    run: RunFn,
    bench_mode: str | None,
    timeout: int,
) -> dict:
    if row.get("status") == "skipped":
        return verify_unavailable(row, "engine skipped")
    if not verify_cmd:
        return verify_unavailable(row, "no --verify-cmd configured")
    if engine in {"codna", "codna-nomem"} and bench_mode != "local":
        return verify_unavailable(row, "Codna inspect-mode patch_ref is not applied to the checkout")
    rc, out, err, dt = run(verify_cmd, cwd=copydir, timeout=timeout)
    row["accuracy_available"] = True
    row["verified"] = rc == 0
    row["verification"] = "pass" if rc == 0 else f"fail ({rc})"
    row["fix_result"] = "verified" if row.get("precheck") == "fail-first" and rc == 0 else (
        "unqualified" if rc == 0 else row["verification"]
    )
    row["verification_time_s"] = round(dt, 1)
    row["verification_notes"] = "" if rc == 0 else last_output_line(out, err, "verify command failed")
    return row
