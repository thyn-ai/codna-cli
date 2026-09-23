"""CI-failure triage for ``check_suite_failure`` fix jobs.

A red check suite on a pull request enqueues a ``codna fix``. Before that job spends anything it
asks three questions, in order, and stops at the first one with a settled answer:

1. **Is the failure still current?** The PR head may have moved, or the suite may have been re-run
   and passed. Then there is nothing to fix on this commit.
2. **Is it a code defect at all?** A ``bun install`` that fails extracting a tarball, a registry
   503, a runner that lost contact, a full disk -- none of these are fixed by editing the
   repository. Left alone, the fix agent ran the repo's tests in its own sandbox, "found" whatever
   failed there, and patched THAT (thyn-ai/algenta#1040, 2026-09-19: a test-only patch, refused
   by the guard, one metered run later). Infrastructure failures end the job ``neutral`` and are
   re-run once when the App may (``actions: write``).
3. **Otherwise it is a code failure**: the failing step and the log around the error travel to
   the agent as the issue text, so it fixes what CI saw and not what its sandbox saw.

Reading job steps and logs needs ``actions: read`` on the installation, which installs made before
2026-09-19 never granted. Without it the job proceeds exactly as before (verdict ``unknown``) and
the check summary asks for the permission. Every network call here is opportunistic and
failure-tolerant: triage can only ever SKIP work, never fail a job.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Verdicts that end the job without running the agent.
TERMINAL = frozenset({"green", "head_moved", "infrastructure"})

PERMISSION_NOTE = (
    "ℹ️ Codna could not read this CI job's steps and log. Grant the Codna GitHub App "
    "**Actions: read** (App settings → Permissions & events, then accept on the installation) so it "
    "can tell an infrastructure failure from a code defect and hand the agent the real CI error; "
    "**Actions: write** also lets it re-run infrastructure failures itself."
)

# Steps whose job is to fetch the world rather than to exercise the code.
_INFRA_STEP_RE = re.compile(
    r"(?i)\b(install|setup|set[ -]up|cache|restore|checkout|check[ -]out|download|fetch|pull|login|"
    r"log[ -]in|bootstrap|provision|prepare|toolchain|dependencies|deps)\b"
)
# What a network, registry or runner failure looks like in a step's log.
_TRANSIENT_RE = re.compile(
    r"(?i)("
    r"fail(?:ed)? extracting tarball|tarball[^\n]*?(?:corrupt|integrity|checksum|extract)|"
    r"integrity check(?:sum)? failed|unexpected end of (?:file|json input|archive)|unexpected eof|"
    r"ECONNRESET|ECONNREFUSED|ETIMEDOUT|EAI_AGAIN|ENOTFOUND|EPIPE|socket hang up|"
    r"connection reset by peer|connection timed out|operation timed out|read timed out|"
    r"tls handshake timeout|temporary failure in name resolution|could not resolve host|"
    r"too many requests|rate limit(?:ed)?(?: exceeded)?|service unavailable|bad gateway|"
    r"gateway time-?out|(?:status(?: code)?|HTTP(?:/[\d.]+)?|error)\s*:?\s*\(?(?:429|50[234])\b|"
    r"unable to (?:download|resolve action)|failed to (?:download|fetch|createartifact|restore)|"
    r"cache service responded with|error response from daemon[^\n]*?(?:pull|manifest|blob|timeout|eof|tls)|"
    r"x509: certificate|no space left on device"
    r")"
)
# Failures of the machine itself, whatever step they surface in.
_RUNNER_LEVEL_RE = re.compile(
    r"(?i)(the runner has received a shutdown signal|lost communication with the server|"
    r"the hosted runner[^\n]*? encountered an error|no space left on device)"
)
# Evidence that the code, not the plumbing, failed -- vetoes the transient reading of an install step.
_CODE_MARKER_RE = re.compile(
    r"(?m)(?:^|\s)(?:FAILED |AssertionError|Traceback \(most recent call last\)|error TS\d{4}|"
    r"SyntaxError:|TypeError:|ReferenceError:|\d+ (?:tests?|specs?) failed|Tests:\s+\d+ failed|"
    r"FAIL:|--- FAIL|panic:|lockfile had changes|frozen-lockfile|lockfile needs to be updated|"
    r"Cannot find module)"
)
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ?")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass
class CITriage:
    verdict: str  # "green" | "head_moved" | "infrastructure" | "code" | "unknown"
    summary: str = ""  # Check Run markdown when the verdict is terminal
    issue_text: str | None = None  # CI evidence for the agent when the verdict is "code"
    note: str | None = None  # appended to the final check summary (a missing permission)
    rerun: bool = False  # the failed jobs were re-run


@dataclass
class _Evidence:
    run: dict
    verdict: str
    step: str | None = None
    window: str | None = None
    info: dict | None = field(default=None)


def focus_log(text: str | None, *, before: int = 60, after: int = 0) -> str | None:
    """The lines around the FIRST ``##[error]`` marker -- the failure that made the job fail;
    later ``if: always()`` steps carry their own -- cleaned of timestamps and colour codes. The
    last ``before`` lines when there is no marker."""
    if not text:
        return None
    lines = [_ANSI_RE.sub("", _TS_RE.sub("", ln)).rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("##[group]", "##[endgroup]"))]
    idx = next((i for i, ln in enumerate(lines) if ln.startswith("##[error]")), None)
    window = lines[-before:] if idx is None else lines[max(0, idx - before): idx + 1 + after]
    return "\n".join(ln[:400] for ln in window)[:8000] or None


def failing_step(info: dict | None) -> str | None:
    """Name of the first step that failed in an Actions job payload."""
    for step in (info or {}).get("steps") or []:
        if isinstance(step, dict) and step.get("conclusion") == "failure":
            return str(step.get("name") or "") or None
    return None


def classify_job(step: str | None, log_window: str | None) -> str:
    """``infrastructure`` / ``code`` / ``unknown`` for one failed Actions job.

    Runner-level failures are infrastructure wherever they surface. An install/setup-style step
    whose log shows a network, registry or archive failure -- and no test/compile failure -- is
    infrastructure too (a lockfile drift in the same step is code). Without a log nothing is
    decided.
    """
    text = log_window or ""
    if _RUNNER_LEVEL_RE.search(text):
        return "infrastructure"
    if not text:
        return "unknown"
    if step and _INFRA_STEP_RE.search(step) and _TRANSIENT_RE.search(text) and not _CODE_MARKER_RE.search(text):
        return "infrastructure"
    return "code"


def _signature(window: str | None) -> str:
    """The first log line that carries the transient/runner signature, for the check summary."""
    for line in (window or "").splitlines():
        if _RUNNER_LEVEL_RE.search(line) or _TRANSIENT_RE.search(line):
            return line.strip()[:160]
    return ""


def _actions_token(job: Any, token: str, github: Any, app_id: str | None,
                   private_key: str | None) -> tuple[str | None, bool]:
    """A token that can read Actions jobs and logs (and re-run them when ``writable``), or None
    when the installation never granted ``actions``. Dev mode (no App auth) tries the ambient
    token and lets the API answer."""
    mint = getattr(github, "installation_token", None)
    if not (job.installation_id and app_id and private_key and callable(mint)):
        return token, bool(token)
    for kind, writable in (("ci_rerun", True), ("ci_triage", False)):
        try:
            minted = mint(app_id, private_key, job.installation_id,
                          repo_full_name=job.repo_full_name, kind=kind)
        except Exception:  # noqa: BLE001 -- 422 "permission not granted" (or anything else): try smaller
            continue
        if minted:
            return str(minted), writable
    return None, False


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 -- triage is opportunistic; a failed lookup is "no evidence"
        return None


def _infrastructure_summary(evidence: list[_Evidence], *, rerun_action: str) -> str:
    lines = ["**codna fix** — not a code defect; nothing was spent.", ""]
    for ev in evidence:
        name = str(ev.run.get("name") or "CI job")
        url = str(ev.run.get("html_url") or "")
        label = f"[{name}]({url})" if url else f"`{name}`"
        step = f" failed in step `{ev.step}`" if ev.step else " failed"
        sig = _signature(ev.window)
        lines.append(f"- {label}{step}" + (f": `{sig}`" if sig else "") + " — an infrastructure error.")
    lines += ["", rerun_action]
    return "\n".join(lines)


def _code_issue_text(job: Any, evidence: list[_Evidence]) -> str:
    parts = [
        f"CI failed on this pull request's head commit {job.ref}. Fix the code so these jobs pass; "
        "do not weaken, skip or delete tests.",
        "",
    ]
    for ev in evidence:
        if ev.verdict != "code":
            continue
        name = str(ev.run.get("name") or "CI job")
        head = f"## {name}" + (f" — failing step: {ev.step}" if ev.step else "")
        parts += [head, "```", ev.window or "(log unavailable)", "```", ""]
    infra = [ev for ev in evidence if ev.verdict == "infrastructure"]
    if infra:
        parts.append("Ignore these jobs, they failed on infrastructure, not code: "
                     + ", ".join(str(ev.run.get("name") or "?") for ev in infra))
    return "\n".join(parts).strip()


def triage(job: Any, token: str, github: Any, *, app_id: str | None = None,
           private_key: str | None = None) -> CITriage:
    """Decide what a ``check_suite_failure`` fix job should do. Never raises."""
    ctx = job.context or {}
    repo = job.repo_full_name

    # 1. Head moved since the failure? The new commit's own CI decides.
    if job.pr_number and job.ref:
        head = _call(getattr(github, "pull_request_head_sha", None), repo, token, job.pr_number)
        if isinstance(head, str) and head and head != job.ref:
            return CITriage("head_moved", summary=(
                f"**codna fix** — superseded. The PR head moved from `{job.ref[:8]}` to `{head[:8]}` "
                "after this CI failure; the new commit's own CI decides whether anything needs fixing. "
                "Nothing was spent."))

    # 2. Suite still red? A re-run replaces its check runs in place.
    suite_id = ctx.get("check_suite_id")
    failing = None
    if isinstance(suite_id, int):
        failing = _call(getattr(github, "failing_check_runs_in_suite", None), repo, token, suite_id)
    if isinstance(failing, list) and not failing:
        return CITriage("green", summary=(
            "**codna fix** — superseded. The failed check suite has no failing job any more (it was "
            "re-run and passed, or a re-run is in progress). Nothing to fix on this commit; nothing was spent."))
    if not isinstance(failing, list):
        return CITriage("unknown")  # cannot see the suite: run as before

    # 3. What failed, and how?
    actions_token, can_rerun = _actions_token(job, token, github, app_id, private_key)
    if not actions_token:
        return CITriage("unknown", note=PERMISSION_NOTE)
    evidence: list[_Evidence] = []
    for run in failing:
        if not isinstance(run, dict):
            continue
        if run.get("app_slug") != "github-actions" or not isinstance(run.get("id"), int):
            evidence.append(_Evidence(run, "unknown"))  # another CI system: no evidence to read
            continue
        info = _call(getattr(github, "actions_job", None), repo, actions_token, run["id"])
        window = focus_log(_call(getattr(github, "actions_job_log_tail", None), repo, actions_token, run["id"]))
        step = failing_step(info if isinstance(info, dict) else None)
        evidence.append(_Evidence(run, classify_job(step, window), step, window,
                                  info if isinstance(info, dict) else None))

    infra = [ev for ev in evidence if ev.verdict == "infrastructure"]
    if evidence and len(infra) == len(evidence):
        run_ids = sorted({ev.info["run_id"] for ev in infra
                          if ev.info and isinstance(ev.info.get("run_id"), int)})
        attempt = max((int(ev.info.get("run_attempt") or 1) for ev in infra if ev.info), default=1)
        rerun = False
        if attempt >= 2:
            action = (f"This was already attempt {attempt} of that workflow run and it failed the same "
                      "way — the runner, registry or network needs a human look, not this pull request.")
        elif not can_rerun or not run_ids:
            action = ("Re-run the failed jobs to continue. Grant the Codna GitHub App **Actions: write** "
                      "to let it re-run infrastructure failures itself.")
        else:
            # A list, not a generator: `all()` on a lazy generator stops at the first refusal and
            # would leave every later workflow run un-re-run while the summary blames "the API".
            outcomes = {rid: bool(_call(getattr(github, "rerun_failed_jobs", None), repo, actions_token, rid))
                        for rid in run_ids}
            done = [str(r) for r, ok in outcomes.items() if ok]
            refused = [str(r) for r, ok in outcomes.items() if not ok]
            rerun = not refused
            if rerun:
                action = (f"Re-ran the failed jobs (workflow run{'s' if len(done) > 1 else ''} "
                          f"{', '.join(done)}, attempt {attempt + 1}).")
            elif done:
                # Say exactly what happened: some runs were re-run, others the API refused.
                action = (f"Re-ran the failed jobs of workflow run{'s' if len(done) > 1 else ''} "
                          f"{', '.join(done)} (attempt {attempt + 1}); the API refused to re-run "
                          f"{', '.join(refused)} -- re-run {'those' if len(refused) > 1 else 'that one'} manually.")
            else:
                action = "Re-running the failed jobs was refused by the API; re-run them manually."
        return CITriage("infrastructure", summary=_infrastructure_summary(infra, rerun_action=action),
                        rerun=rerun)
    if any(ev.verdict == "code" for ev in evidence):
        return CITriage("code", issue_text=_code_issue_text(job, evidence))
    return CITriage("unknown")
