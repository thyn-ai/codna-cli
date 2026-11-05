"""The two goal-mode activities: the CI gate and episode recall (D5).

Both live here rather than in `activity_steps.py` because `activity_steps.py` is at the
repository's 1000-line ceiling and because they answer a DIFFERENT question from every other step:
the other steps transform one task's state, these answer run-level questions — "is the pull request
green?" and "what did the earlier tasks actually achieve?".

**`watch_ci` reuses the CI triage policy rather than re-deriving it.** `webhook_ci_triage` already
decides "is this a code defect, or plumbing?" for the App's fix jobs; a second, divergent answer on
the durable path would make the two disagree about the same red check. The required-check READ is
`review_github.failing_required_checks`, the one the PR card already uses.

**`recall` reads episode FILES, and verifies their digests.** A run that carried episode SUMMARY
TEXT in its input would grow with the run, and could be replayed into a plan the ledger never
recorded. Reading by reference means a retried activity recalls the same bytes (or fails loudly if
the file was tampered with), and the digest recorded at `remember_episode` time is what is checked.

The SHAPES live in `goal_contracts.py` (a leaf) because workflow code imports them; this module
imports `activity_support`, which reaches git and subprocess, and must never be imported by a
workflow.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

from .goal import MAX_EPISODES, recap_from_payload
from .goal_contracts import (
    MAX_EPISODE_BYTES,
    NAME_RECALL,
    RecallOutcome,
    RecallRequest,
    RecalledEpisode,
    WatchCiOutcome,
    WatchCiRequest,
)

__all__ = ["NAME_RECALL", "recall_episodes", "watch_ci"]


class _EpisodeUnreadable(Exception):
    """One episode file cannot be used, for a reason worth recording (never worth failing a run)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# -- recall ------------------------------------------------------------------------------------


def recall_episodes(request: RecallRequest) -> RecallOutcome:
    """Read this run's episode files by reference and return bounded recaps.

    Deterministic in the FILES it is given: the same run directory produces the same recaps and the
    same order (oldest first, capped to the newest `MAX_EPISODES`). A digest mismatch excludes the
    file from the recaps AND records it as missing — planning on a file that is not the episode the
    run recorded would be planning on something nobody audited, and silently dropping it would look
    like a task that never ran.
    """
    from .activity_steps import _runtime  # noqa: PLC2701 - the ONE runtime accessor

    runtime = _runtime()
    request.validate()
    refs = request.refs
    if len(refs) > MAX_EPISODES:
        # Newest win: the plan for task N is about what tasks N-1 and N-2 learned, not task 3.
        refs = refs[-MAX_EPISODES:]
    run_dir = _run_dir(runtime, request.run_id)
    episodes: list[RecalledEpisode] = []
    missing: list[str] = []
    for ref in refs:
        task_id = str(ref.get("task_id") or "")
        try:
            record = _read_episode(run_dir, str(ref.get("path") or ""), str(ref.get("digest") or ""))
        except _EpisodeUnreadable as exc:
            missing.append(f"{task_id or '?'}: {exc.reason}")
            continue
        except (ValueError, OSError) as exc:
            # A file that is present but not decodable is the same class of answer as a missing one:
            # this run's plan proceeds without that episode, and the ledger says why.
            missing.append(f"{task_id or '?'}: unreadable ({exc})")
            continue
        recap = recap_from_payload(record)
        episodes.append(
            RecalledEpisode(
                task_id=recap.task_id or task_id,
                outcome=recap.outcome,
                brief=recap.brief,
                verify_passed=recap.verify_passed,
                review_verdict=recap.review_verdict,
                pr_url=recap.pr_url,
                plan_ref=recap.plan_ref,
                digest=str(ref.get("digest") or ""),
            )
        )
    runtime.ledger(request.run_id).append(
        "note",
        task_id=None,
        payload={
            "event": "episodes.recalled",
            "count": len(episodes),
            "tasks": [episode.task_id for episode in episodes],
            "missing": list(missing),
        },
    )
    return RecallOutcome(episodes=tuple(episodes), missing=tuple(missing))


def _read_episode(run_dir: Path, path: str, digest: str) -> dict[str, Any]:
    """One episode file, digest-verified, or a reason it cannot be used."""
    if not path or not digest:
        raise _EpisodeUnreadable("no path or digest")
    # The path is `episodes/<task>.json`, relative to the run directory — never absolute, never
    # escaping it. A reference that could point outside the run directory would let a plan read an
    # arbitrary file, so it is refused rather than resolved.
    candidate = (run_dir / path).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise _EpisodeUnreadable("path escapes the run directory") from exc
    if not candidate.is_file():
        raise _EpisodeUnreadable("file is gone")
    if candidate.stat().st_size > MAX_EPISODE_BYTES:
        raise _EpisodeUnreadable("file is larger than an episode may be")
    body = json.loads(candidate.read_bytes().decode("utf-8"))
    if not isinstance(body, dict):
        raise _EpisodeUnreadable("file does not decode to an object")
    if str(body.get("digest") or "") != digest:
        raise _EpisodeUnreadable("digest does not match the recorded episode")
    # Re-derive the digest from the body, so a file whose inner `digest` was edited is caught too.
    canonical = json.dumps(
        {key: value for key, value in body.items() if key != "digest"},
        sort_keys=True,
        separators=(",", ":"),
    )
    if hashlib.sha256(canonical.encode()).hexdigest() != digest:
        raise _EpisodeUnreadable("episode body does not hash to its recorded digest")
    return body


def _run_dir(runtime: Any, run_id: str) -> Path:
    from .config import validate_run_id

    directory = Path(runtime.runs_root) / validate_run_id(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# -- the CI gate -------------------------------------------------------------------------------


def watch_ci(request: WatchCiRequest) -> WatchCiOutcome:
    """Read the required checks on a PR's head and classify what is red.

    The three answers that matter, kept distinct:

    * ``passed`` — no required check is failing. The only state that satisfies a CI gate.
    * ``failed`` — a required check failed its steps. This is the one that earns a "fix CI" task.
    * ``infrastructure`` — a required check died without running (a timeout, a startup failure, or a
      fetch/setup-shaped check). A fix task would patch code CI never judged; the run reports it
      and does not spend a task on it.

    ``pending`` and ``unknown`` are also distinct: a check still running is not green (the caller
    keeps waiting), and "we could not read the rollup" is not green either — a missing token must not
    let a run declare its goal met.
    """
    from .activity_steps import _runtime  # noqa: PLC2701 - the ONE runtime accessor

    runtime = _runtime()
    request.validate()
    runtime.beat({"phase": "watch_ci", "pr_number": request.pr_number})
    seam = getattr(runtime, "watch_ci", None)
    if seam is not None:
        # The test seam. It is consulted FIRST so a test drives the loop through the real body's
        # ledger and classification, not through a stub that replaces them.
        outcome = seam(request)
        outcome.validate()
        _record(runtime, request, outcome)
        return outcome
    outcome = _read_ci(runtime, request)
    _record(runtime, request, outcome)
    return outcome


def _read_ci(runtime: Any, request: WatchCiRequest) -> WatchCiOutcome:
    """The real read: GitHub's required checks on this PR's head, classified by the shared policy."""
    from .. import review_github

    from .activity_support import git_ok
    from .goal import ci_verdict

    token = runtime.github_token() if runtime.github_token is not None else _env_github_token()
    if not token:
        return WatchCiOutcome(
            state="unknown",
            reason="no GitHub token: required checks could not be read",
            head_sha=request.head_sha,
        )
    origin = (git_ok(runtime, request.repo_dir, ["config", "--get", "remote.origin.url"]) or "").strip()
    slug = review_github.repo_slug_from_url(origin)
    if not slug:
        return WatchCiOutcome(
            state="unknown",
            reason="the repository has no parseable GitHub origin",
            head_sha=request.head_sha,
        )
    red = review_github.failing_required_checks(slug, request.pr_number, token)
    if red is None:
        return WatchCiOutcome(
            state="unknown",
            reason="the required-check rollup could not be read",
            head_sha=request.head_sha,
        )
    state = ci_verdict(red)
    checks = tuple(
        (str(check.get("name") or "?"), str(check.get("state") or "")) for check in red
    )
    head_sha = str(red[0].get("head_sha") or "") if red else request.head_sha
    if state == "failed" and request.head_sha and head_sha and head_sha != request.head_sha:
        # The PR moved under us: these checks describe a commit this run did not push, so "failed"
        # would send a fix task after work somebody else is doing. `pending` is the honest answer —
        # not green, and not this run's failure.
        return WatchCiOutcome(
            state="pending",
            checks=checks,
            head_sha=head_sha,
            reason="the pull request head moved since this run pushed",
        )
    return WatchCiOutcome(state=state, checks=checks, head_sha=head_sha)


def _record(runtime: Any, request: WatchCiRequest, outcome: WatchCiOutcome) -> None:
    """One ledger note per read: the CI TRANSITION is an audit fact, not an implementation detail."""
    runtime.ledger(request.run_id).append(
        "note",
        task_id=None,
        payload={
            "event": "ci.watched",
            "pr_number": request.pr_number,
            "state": outcome.state,
            "checks": [list(entry) for entry in outcome.checks],
            "head_sha": outcome.head_sha,
            "reason": outcome.reason,
        },
    )


def _env_github_token() -> str | None:
    for name in ("CODNA_GITHUB_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


_ = (Final,)
