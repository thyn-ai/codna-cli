"""The CI-fix loop: read the pull request's checks, and while red spend a bounded fix attempt.

── WHY THIS IS ITS OWN MODULE ──────────────────────────────────────────────────────────────────

`run_workflow.py` is held under this package's 1000-line modularity ceiling
(`tests/test_modularity.py`). The stop-delivery work pushed it over, and the ceiling exists to force
a split before a module sprawls — so this cohesive concern moved out rather than the number moving up.

── THE TWO INDEPENDENT EXITS, WHICH IS THE WHOLE DESIGN ────────────────────────────────────────

`max_ci_attempts` bounds how many fix tasks one red PR may spend; the RUN's budget bounds how much
money it may spend. Either alone stops the loop: an attempt cap with no budget cap would spend the
whole run on one PR, and a budget cap with no attempt cap would keep buying attempts until the run
died. Both are checked before every fix task.

`infrastructure`, `pending` and `unknown` never start a fix task. The first two are not this run's
code; the third means the rollup could not be read, and patching code to satisfy a check we could not
see is the failure `webhook_ci_triage` exists to prevent.

A `failed` state with no attempts left is not an error: the state stays `failed`, `evaluate_stop`
reports `ci=failed`, and `_ci_stop` says WHICH bound ended the loop — the difference between "we tried
three times" and "we ran out of money".

No clock, no I/O, no randomness: importable from the Temporal sandbox. The workflow state it reads
(`_spec`, `_ci_state`, `_ci_attempts`, `_ci_stop`, `_completed`, `_spent`, `_episodes`) is declared
here so a subclass cannot silently omit one.
"""

from __future__ import annotations

from typing import Any

from .contracts import NAME_WATCH_CI, TaskSpec, TaskCompletion, WatchCiRequest
from .goal import build_ci_fix_brief
from .workflow_support import _pr_coordinates, run_budget_left, workflow_activity


class CiFixLoop:
    """The bounded CI repair loop, mixed into the run workflow."""

    # Declared so the mixin is explicit about what it reads from the run.
    _spec: Any
    _ci_state: str | None
    _ci_attempts: int
    _ci_stop: str | None
    _completed: list[Any]
    _spent: float
    _episodes: list[Any]

    async def _ci_gate(self, task: TaskSpec, child: TaskCompletion) -> None:
        """Read CI, and while it is red start ONE fix task and read again — bounded twice over.

        **This is the loop, and it has two independent exits.** ``max_ci_attempts`` bounds how many
        fix tasks one red PR may spend; the RUN's budget bounds how much money it may spend. Either
        alone stops it: an attempt cap with no budget cap would spend the whole run on one PR, and a
        budget cap with no attempt cap would keep buying attempts until the run died. Both are
        checked before every fix task, so the loop cannot outlive either.

        A ``failed`` CI state with no attempts left is not retried and not an error: the state stays
        ``failed``, `evaluate_stop` reports ``ci=failed``, and ``self._ci_stop`` says WHICH bound
        ended the loop — the difference between "we tried three times" and "we ran out of money".

        ``infrastructure``, ``pending`` and ``unknown`` never start a fix task. The first two are not
        this run's code; the third means the rollup could not be read, and patching code to satisfy a
        check we could not see is the failure `webhook_ci_triage` exists to prevent.
        """
        spec = self._spec
        assert spec is not None and spec.goal is not None
        while True:
            pr_number, head_sha = _pr_coordinates(child)
            if pr_number is None:
                # No pull request on this task, so there is no head for CI to judge. Leaving the state
                # alone is deliberate: `evaluate_stop` then reports `ci=failed`, the honest answer for
                # a goal that requires a green PR and has none.
                return
            outcome = await workflow_activity(
                NAME_WATCH_CI,
                WatchCiRequest(
                    run_id=spec.run_id,
                    repo_dir=spec.repo_dir,
                    pr_number=pr_number,
                    head_sha=head_sha,
                    base_branch=spec.base_branch,
                ),
            )
            self._ci_state = outcome.state
            if outcome.state != "failed":
                return
            if self._ci_attempts >= spec.goal.max_ci_attempts:
                self._ci_stop = (
                    f"ci still red after {self._ci_attempts} attempt(s) "
                    f"(max_ci_attempts={spec.goal.max_ci_attempts})"
                )
                await self._notify(
                    "note",
                    f"run {spec.run_id}: CI still red on PR #{pr_number} after "
                    f"{self._ci_attempts} fix attempt(s); not starting another "
                    f"(max_ci_attempts={spec.goal.max_ci_attempts})",
                )
                return
            budget_left = run_budget_left(spec, self._spent)
            if budget_left is not None and budget_left <= 0:
                self._ci_stop = (
                    f"ci still red and the run budget is exhausted "
                    f"({self._ci_attempts} attempt(s) of {spec.goal.max_ci_attempts})"
                )
                await self._notify(
                    "note",
                    f"run {spec.run_id}: CI fix loop stopped by the run budget, not by the attempt cap",
                )
                return
            child = await self._run_ci_fix(
                task, child, pr_number=pr_number, head_sha=head_sha, checks=outcome.checks,
                remaining_usd=budget_left,
            )

    async def _run_ci_fix(
        self,
        task: TaskSpec,
        child: TaskCompletion,
        *,
        pr_number: int,
        head_sha: str | None,
        checks: Any,
        remaining_usd: float | None,
    ) -> TaskCompletion:
        """Start ONE "fix CI" task on the same branch and consolidate its result into the run.

        The fix task is a REAL child task: its own plan/implement/verify/review/PR steps, against the
        same worktree and the same pull request. That is what makes the next CI read describe the
        fix — anything cheaper (a turn with a synthetic prompt) would not produce a commit on the
        branch the check suite is running against.
        """
        spec = self._spec
        assert spec is not None and spec.goal is not None
        attempt = self._ci_attempts + 1
        self._ci_attempts = attempt
        brief = build_ci_fix_brief(
            pr_number=pr_number, head_sha=head_sha, checks=checks, attempt=attempt
        )
        fix = task.with_ci_fix(
            task_id=f"{task.task_id}-ci{attempt}", brief=brief, goal_context=task.goal_context
        )
        try:
            completion = await self._run_task(fix, remaining_usd=remaining_usd)
        except Exception as exc:  # noqa: BLE001 - a failed fix attempt is an ANSWER, not a crash
            # A fix that cannot even complete (its turn failed, its verify blew up) must not turn the
            # RUN red: the goal is merely still unmet, and the CI loop's bound decides whether to try
            # again. Escaping here would lose the run's whole result over one repair attempt — and the
            # attempt has already been counted, so the loop cannot spin on it.
            await self._notify(
                "note",
                f"run {spec.run_id}: fix CI attempt {attempt} FAILED "
                f"({type(exc).__name__}) on PR #{pr_number}; counting it against max_ci_attempts",
            )
            return child
        self._completed.append(completion.summary())
        self._spent += completion.cost_usd
        episode = await self._recall_episode_ref(fix, completion)
        if episode is not None:
            self._episodes.append(episode)
        await self._notify(
            "note",
            f"run {spec.run_id}: fix CI attempt {attempt} finished ({completion.outcome}) "
            f"on PR #{pr_number}",
        )
        return completion
