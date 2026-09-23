"""Check Run / reply summaries for webhook jobs: the one-line, markdown-safe text a job leaves
behind, and the CLI-error parsing that produces it.

Split out of webhook_worker.py on 2026-09-18 (the 1000-line modularity ceiling); pure functions,
no I/O. Names are the lowercase Check Run names ("codna fix failed (cli_error): ...") -- see
webhook.check_run_name.
"""
from __future__ import annotations

import json
import re

from .webhook import check_run_name

# Case-insensitive on purpose: summaries were briefly Title Case ("Codna Fix failed (...)") in 2026-09.
_CLI_SUMMARY = re.compile(r"^codna \w+ failed \(([^)]+)\): (.*)$", re.S | re.IGNORECASE)


# CLI error codes whose failure is a property of the inputs, not of the moment.
_NON_RETRYABLE_CODES = frozenset({"cli_error", "usage_error", "invalid_input", "input_error"})

# CLI error codes that mean "this sandbox cannot run the repository's tests" (testrun.
# TestEnvironmentUnavailable): a property of the environment, not of the commit, so the check ends
# neutral -- nothing to fix here, nothing to retry -- the way an infrastructure CI failure does.
_NEUTRAL_CODES = frozenset({"test_environment_unavailable"})


_SUMMARY_CODE = re.compile(r"^codna \w+ failed \(([^)]+)\)", re.IGNORECASE)
_SUMMARY_PARTS = re.compile(r"^(codna \w+) failed \(([^)]+)\): (.*)$", re.S | re.IGNORECASE)


def _summary_is_neutral(summary: str) -> bool:
    m = _SUMMARY_CODE.match(summary or "")
    return bool(m and m.group(1) in _NEUTRAL_CODES)


def _neutral_check_summary(summary: str) -> str:
    """'codna fix failed (test_environment_unavailable): <why + how to configure>' -> the Check Run
    text for a job that was skipped, not failed. The CLI's structured-detail block (the runner's
    output tail) stays in the logs."""
    m = _SUMMARY_PARTS.match(summary or "")
    name, message = (m.group(1).lower(), m.group(3)) if m else ("codna fix", summary or "")
    return f"**{name}** — skipped: {message.split(_DETAILS_FENCE, 1)[0].strip()}\n\nNo fix was attempted."

# Structured-error fields worth showing under the sentence. `raw_head` / `raw_tail` are the review
# agent's actual (redacted) output when it was not findings JSON: without them "did not return
# parseable findings JSON" names a symptom and withholds the diagnosis (thyn-ai/test-codna-app-e2e#15,
# 2026-09-19: three identical red runs, nothing to read). The agent-core fields (`status`,
# `terminal_state`, `sidecar_url`, `error`) distinguish a stub runtime from a dead sidecar.
_DETAIL_KEYS = ("cause_code", "status", "terminal_state", "error", "sidecar_url", "raw_chars", "raw_head", "raw_tail")
_DETAILS_FENCE = "\n\n```\n"
_DETAIL_VALUE_CHARS = 600


def _summary_is_retryable(summary: str) -> bool:
    m = _SUMMARY_CODE.match(summary or "")
    return not (m and m.group(1) in _NON_RETRYABLE_CODES)


def _terse_cli_failure(summary: str) -> str:
    """One readable clause from the CLI's structured error, minus the operator payload.

    'codna fix failed (cli_error): apply failed: Git command failed ... | code=git_command_failed |
    details={"args": [...], "cwd": "/tmp/...", "stderr": "error: corrupt patch at line 99"} (patch
    ref: ...)' -> 'apply failed: Git command failed ... (git: error: corrupt patch at line 99)
    [git_command_failed]'. Paths and argv stay in the Check Run summary / logs, not in the thread.
    """
    m = _CLI_SUMMARY.match(summary or "")
    if not m:
        return summary
    code, rest = m.group(1), m.group(2)
    rest = rest.split(_DETAILS_FENCE, 1)[0]  # the structured-detail block stays in the Check Run
    rest = re.sub(r"\s*\(patch ref: [^)]*\)?\s*$", "", rest)
    # The details payload may be TRUNCATED (the summary is capped), so never require a closing
    # brace: cut at the marker and pull stderr out with a regex that tolerates a missing end quote.
    details_text = ""
    dm = re.search(r"\s*\|\s*details=", rest)
    if dm:
        details_text, rest = rest[dm.end():], rest[: dm.start()]
    cm = re.search(r"\s*\|\s*code=([\w.-]+)\s*$", rest)
    if cm:
        code, rest = cm.group(1), rest[: cm.start()]
    hint = ""
    sm = re.search(r'"stderr":\s*"((?:[^"\\]|\\.)*)"?', details_text, re.S)
    if sm:
        stderr = sm.group(1).encode().decode("unicode_escape", errors="replace")
        lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
        # the reason is usually the LAST informative line ("! [remote rejected] ... (why)"), not
        # the "To https://..." banner git prints first
        strong = [ln for ln in lines if re.search(r"rejected|refusing|denied|GH0\d\d", ln, re.I)]
        weak = [ln for ln in lines if re.search(r"error|fatal", ln, re.I)]
        pick = (strong or weak or lines)[-1] if lines else ""
        if pick:
            hint = f" (git: {pick[:200]})"
    return f"{rest.strip().rstrip('.')}{hint} [{code}]"


def _review_job_summary(stdout: str) -> str | None:
    """A short, markdown-safe summary for a review job's wrapper Check Run.

    ``codna review ... --json`` (the only command codna_command adds ``--json`` to) prints a
    findings-review result dict as its ENTIRE stdout -- the same shape
    ``cli/codna/review.py::render_findings`` renders for a human. That renderer is for a
    terminal, though: several of its lines are indented 4+ spaces, which GitHub's markdown
    turns into a code block rather than the intended text, so it is deliberately not reused
    here. Returns None (never raises) on anything that isn't exactly this shape, so the caller
    falls back to its existing raw-stdout-tail behavior instead of guessing.
    """
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(result, dict) or "conclusion" not in result or "findings" not in result:
        return None
    findings = result.get("findings") or []
    if not findings:
        note = result.get("note")
        approved = " · approved" if (result.get("posted") or {}).get("event") == "APPROVE" else ""
        return f"codna review: {note}{approved}" if note else f"codna review: no high-confidence issues found ✅{approved}"
    posted = result.get("posted") or {}
    inline = posted.get("inline_posted", 0)
    summary_n = result.get("summary_count", 0)
    approved = " · approved" if posted.get("event") == "APPROVE" else ""
    return (
        f"codna review: {len(findings)} finding(s) — {inline} inline, {summary_n} in summary "
        f"(conclusion: {result.get('conclusion', 'unknown')}){approved}. See the PR's review comments for details."
    )


def _cli_completed_check(stdout: str, check_run_id: int | None) -> bool:
    """True when ``codna review --post`` reports it completed THIS job's Check Run itself."""
    if check_run_id is None:
        return False
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    check = ((result.get("posted") or {}).get("check") or {}) if isinstance(result, dict) else {}
    return bool(check.get("updated_existing")) and check.get("id") == check_run_id



def _details_block(details: object) -> str:
    """The CLI's structured error details, bounded and fenced, or "" when there is nothing to show.
    Values were redacted at the source (cline_agent._unparseable_output_details); this only bounds
    them and keeps a stray fence from breaking out of the block."""
    if not isinstance(details, dict):
        return ""
    lines = []
    for key in _DETAIL_KEYS:
        value = details.get(key)
        if value in (None, "", [], {}):
            continue
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        # Normalise line endings BEFORE the cut: a cut between "\r" and "\n" left a bare "\r".
        text = text.replace("```", "'" * 3).replace("\r\n", "\n").replace("\r", "\n")
        if len(text) > _DETAIL_VALUE_CHARS:
            text = text[:_DETAIL_VALUE_CHARS] + "…"
        # A multi-line value (raw_head is prose, often several paragraphs) keeps one entry per key:
        # continuation lines are indented under it instead of masquerading as the next `key:`.
        text = text.replace("\n", "\n    ")
        lines.append(f"{key}: {text}")
    return _DETAILS_FENCE + "\n".join(lines) + "\n```" if lines else ""


def _error_json_summary(kind: str, *streams: str | None) -> str | None:
    """Short, markdown-safe summary when the CLI ended with its documented structured error.

    ``codna`` prints ``{"error": {"code": ..., "message": ...}}`` (indent=2) to stderr on every
    CodnaError (cli.py), and the generic tail-of-output fallback pasted that JSON verbatim into
    the "codna fix" Check Run summary -- two algenta runs on 2026-09-17 read exactly like that.
    Finds the LAST such object across the given streams (stderr is searched first by passing it
    last) and renders its message as one sentence. None when there is no such object.
    """
    decoder = json.JSONDecoder()
    for text in reversed([s or "" for s in streams]):
        pos = text.rfind('"error"')
        while pos >= 0:
            start = text.rfind("{", 0, pos)
            if start < 0:
                break
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except ValueError:
                obj = None
            err = obj.get("error") if isinstance(obj, dict) else None
            if isinstance(err, dict) and err.get("message"):
                code = err.get("code") or "error"
                # 1500, not 400: the CLI embeds git's stderr at the END of the message, and a
                # 400-char cap cut every live push/apply failure right before its reason.
                return (f"{check_run_name(kind)} failed ({code}): {str(err['message'])[:1500]}"
                        + _details_block(err.get("details")))
            pos = text.rfind('"error"', 0, pos)
    return None
