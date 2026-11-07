"""The RUN contracts — what a run is and what it reports.

Split from `contracts.py` where the module crossed the repository's 1000-line ceiling
(`test_modularity.py`), on the natural boundary: a TASK spec/results pair lives in `contracts.py`
because a task is the unit a workflow executes; a RUN spec/results pair lives here because a run is
the unit that OWNS tasks and survives continue-as-new.

That ownership is the reason these types are shaped the way they are:

* `RunSpec.tasks` is the whole ordered task list, so a run can be resumed by a new execution with no
  external store — the tasks travel in the workflow input.
* `RunProgress` is what crosses a continue-as-new boundary, and it is deliberately small and
  fixed-size: a completed `TaskSummary` per task, a spend total and a generation counter. It must be,
  because it becomes the next execution's input.
* `RunSummary`/`TaskSummary` carry a PR URL and an outcome, not a `TaskState`: what a run reports is
  what a human reads, and a run that accumulated every task's full state would grow without bound.

Everything here is re-exported from `contracts`, so callers keep importing from one module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from .contract_primitives import (
    ContractError,
    GraphEngine,
    optional_cap,
    require_engine,
    require_int,
    require_ref,
)
# `TaskSpec`/`TaskSummary`/`GRAPH_ENGINES` are re-exported by `contracts`, which imports THIS module at
# the end of its own body. Importing them at module scope here would close that cycle, so they are
# resolved at call time from the module that owns them. Type checking still sees them via TYPE_CHECKING.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .contracts import TaskSpec
    from .goal import EpisodeRef, GoalSpec

__all__ = [
    "RunCompletion",
    "RunOutcome",
    "RunProgress",
    "RunSpec",
    "TaskCompletion",
    "TaskSummary",
]

RunOutcome = Literal["completed", "budget_exhausted", "stopped", "goal_unmet"]


@dataclass(frozen=True)
class TaskSummary:
    """A task result small enough to survive a continue-as-new round trip.

    Five scalars and two optional strings — the largest thing `RunProgress` may carry per task, and
    therefore the thing that keeps a long run's next-execution input bounded.

    ``verify_passed``/``review_verdict`` are D5 goal mode's EVIDENCE: the two facts the stop
    condition is evaluated over. They are read from the task's own state (not from its episode file),
    so the run can decide "is the goal met" without an I/O call in workflow code. Absent in a payload
    recorded before goal mode, hence ``None`` — which `evaluate_stop` treats as "not proven", the
    only safe reading of a check that never ran.
    """

    task_id: str
    outcome: str
    cost_usd: float
    turn_seq: int
    approved_by: str | None = None
    pr_url: str | None = None
    verify_passed: bool | None = None
    review_verdict: str | None = None

    @property
    def unpublished(self) -> bool:
        """Whether this task finished its work WITHOUT publishing it (nothing to publish to).

        A property of the OUTCOME rather than of the state, because `TaskSummary` is what a run keeps
        across a continue-as-new boundary: reading the state here would work on the first generation
        and silently stop working after a restart, which is the one place the distinction must not
        be lost.
        """
        return self.outcome == "approved_unpublished"


@dataclass(frozen=True)
class RunSpec:
    """A run: an ordered set of tasks against one checkout.

    **Goal mode (D5)** is the same run plus three additive fields: ``goal`` (the objective and its
    stop condition), ``episodes`` (one reference per completed task, so a plan boundary can recall
    what earlier tasks achieved) and ``pr_number``/``ci_state`` bookkeeping carried across
    continue-as-new. All of them default to "no goal", which is what makes a task-mode run's history
    and behaviour byte-identical to what it was before goal mode existed.
    """

    run_id: str
    repo_dir: str
    base_branch: str
    tasks: "tuple[TaskSpec, ...]"
    max_cost_usd: float | None = None
    engine: GraphEngine = "langgraph"
    # The soak profile's pacing knob. Zero in normal use; time-skipping makes it instant in tests.
    idle_between_tasks_s: float = 0.0
    continue_as_new_tasks: int = 10
    continue_as_new_history_events: int = 5000
    schema_version: int = 1
    # D5: whether this run gives each task its own worktree. Carried at the RUN level as well as on
    # each `TaskSpec` because the clean-up decision is a property of the run: a run that opted in is
    # the only one that may sweep ``<runs_root>/<run>/``, and it must sweep it even if a task's own
    # spec was built by an older caller that never set the flag.
    worktree: bool = False
    # D5 GOAL MODE. ``None`` means "a task-mode run": the goal loop, the CI gate and the plan-side
    # episode recall are all skipped, so a spec recorded before this field existed decodes to exactly
    # the run it meant.
    goal: "GoalSpec | None" = None
    # One episode reference per COMPLETED task, oldest first. References only — the text stays in the
    # run directory, and `goal_steps.recall` re-reads it per plan boundary.
    episodes: tuple["EpisodeRef", ...] = ()
    # The CI gate's state for this run, carried in the input so a continued-as-new execution resumes
    # the same loop rather than re-reading CI from the top. ``None`` = not looked at yet.
    ci_state: str | None = None
    # How many "fix CI" tasks this run has already started. A BOUND, independent of the task list:
    # a CI loop repeats on one PR, so counting it against `max_tasks` would silently authorise one
    # fix attempt per task the goal had left.
    ci_attempts: int = 0
    # D7/MULTITASK: how many tasks of this run may be IN FLIGHT at once. `1` is the historical
    # behaviour (each child awaited before the next starts) and stays the default, so a spec recorded
    # before this field existed replays as the strictly sequential run it was.
    #
    # A bound rather than "all at once": a run is a list of briefs against ONE repository (or one
    # worktree each), and unbounded parallelism would start thirty paid turns in a burst. `1` also
    # keeps the shared-checkout case correct, since two tasks cannot work in one tree simultaneously
    # — `validate()` requires either `worktree` or `max_parallel_tasks == 1`.
    max_parallel_tasks: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError(
                "unsupported run spec schema version",
                code="unsupported_contract_schema",
                details={"schema_version": self.schema_version},
            )
        require_ref(self.run_id, "run_id")
        if not isinstance(self.repo_dir, str) or not self.repo_dir.startswith("/"):
            raise ContractError(
                "repo_dir must be an absolute path",
                code="contract_invalid",
                details={"field": "repo_dir"},
            )
        require_ref(self.base_branch, "base_branch")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ContractError(
                "a run needs at least one task", code="contract_invalid", details={"field": "tasks"}
            )
        from .contracts import TaskSpec

        seen: set[str] = set()
        for task in self.tasks:
            if not isinstance(task, TaskSpec):
                raise ContractError(
                    "tasks must be TaskSpec values",
                    code="contract_invalid",
                    details={"field": "tasks"},
                )
            task.validate()
            if task.run_id != self.run_id:
                raise ContractError(
                    "every task must belong to the run it is started under",
                    code="contract_invalid",
                    details={"run_id": self.run_id, "task_run_id": task.run_id},
                )
            if task.task_id in seen:
                raise ContractError(
                    "task ids must be unique within a run",
                    code="contract_invalid",
                    details={"task_id": task.task_id},
                )
            seen.add(task.task_id)
        optional_cap(self.max_cost_usd, "max_cost_usd")
        require_engine(self.engine)
        if not isinstance(self.idle_between_tasks_s, (int, float)) or self.idle_between_tasks_s < 0:
            raise ContractError(
                "idle_between_tasks_s must be a non-negative number",
                code="contract_invalid",
                details={"field": "idle_between_tasks_s"},
            )
        require_int(self.continue_as_new_tasks, "continue_as_new_tasks", minimum=1)
        require_int(self.continue_as_new_history_events, "continue_as_new_history_events", minimum=100)
        if not isinstance(self.worktree, bool):
            raise ContractError(
                "worktree must be a boolean",
                code="contract_invalid",
                details={"field": "worktree", "value": repr(self.worktree)},
            )
        self._validate_goal()

    def _validate_goal(self) -> None:
        """The goal fields: a spec, the episode references, and the CI loop's own bound.

        Checked together because they are one feature: a goal run without a stop condition would
        never be done, and episodes without a goal are references nothing recalls.
        """
        from .goal import EpisodeRef as _EpisodeRef
        from .goal import GoalSpec as _GoalSpec

        if self.goal is not None:
            if not isinstance(self.goal, _GoalSpec):
                raise ContractError(
                    "goal must be a GoalSpec",
                    code="contract_invalid",
                    details={"field": "goal", "type": type(self.goal).__name__},
                )
            self.goal.validate()
        if not isinstance(self.episodes, tuple):
            raise ContractError(
                "episodes must be a tuple of EpisodeRef values",
                code="contract_invalid",
                details={"field": "episodes"},
            )
        for episode in self.episodes:
            if not isinstance(episode, _EpisodeRef):
                raise ContractError(
                    "episodes must be a tuple of EpisodeRef values",
                    code="contract_invalid",
                    details={"field": "episodes", "type": type(episode).__name__},
                )
            episode.validate()
        if self.ci_state is not None:
            from .goal import CI_STATES

            if self.ci_state not in CI_STATES:
                raise ContractError(
                    f"ci_state must be null or one of {', '.join(CI_STATES)}",
                    code="contract_invalid",
                    details={"field": "ci_state", "value": repr(self.ci_state)},
                )
        require_int(self.ci_attempts, "ci_attempts", minimum=0)
        require_int(self.max_parallel_tasks, "max_parallel_tasks", minimum=1)
        if self.max_parallel_tasks > 1 and self.goal is not None:
            # The goal loop is a sequence of BOUNDARIES: each verdict is computed over "every task so
            # far" and each CI gate reads the run's pull request. Two tasks crossing one boundary would
            # produce a verdict describing a state that never existed, so the combination is refused
            # here rather than silently serialised (which would make `--parallel` a silent no-op).
            raise ContractError(
                "goal mode is sequential: max_parallel_tasks must be 1",
                code="contract_invalid",
                details={"field": "max_parallel_tasks", "conflicts_with": "goal"},
            )
        if self.max_parallel_tasks > 1 and not self.worktree:
            # Fail closed on the tree, not on the idea: two tasks in ONE checkout would edit, commit,
            # verify and review the same files at the same time — the result is whichever write landed
            # last, with a ledger that claims two tasks were done. A worktree per task is what makes
            # concurrent tasks meaningful, so parallelism REQUIRES it (and the pairwise requirement is
            # stated rather than assumed).
            raise ContractError(
                f"max_parallel_tasks={self.max_parallel_tasks} needs worktree=true: concurrent tasks "
                "would otherwise share one checkout",
                code="contract_invalid",
                details={"field": "max_parallel_tasks", "requires": "worktree"},
            )

    def remaining(self, completed: tuple[TaskSummary, ...]) -> "tuple[TaskSpec, ...]":
        """The tasks not yet done, in order — keyed by id, never by position.

        A run whose task ids repeat must not skip a different task because a counter said so.
        """
        done = {summary.task_id for summary in completed}
        return tuple(task for task in self.tasks if task.task_id not in done)


@dataclass(frozen=True)
class RunProgress:
    """The state `RunWorkflow` carries across a continue-as-new boundary.

    Small and fixed-size on purpose: it becomes the NEXT execution's input, so it must not grow with
    the run. One small summary per completed task is the most it can hold — plus, in goal mode, one
    EPISODE REFERENCE per completed task (two short strings, never a summary), the CI gate's last
    verdict and how many fix attempts have been spent.
    """

    completed: tuple[TaskSummary, ...] = ()
    spent_usd: float = 0.0
    generations: int = 1
    outcome: RunOutcome = "completed"
    # D5 goal mode. Empty/None for a task-mode run, which is what keeps an in-flight run's next
    # execution input identical to what it was before this feature existed.
    episodes: tuple["EpisodeRef", ...] = ()
    ci_state: str | None = None
    ci_attempts: int = 0
    # Why a goal run stopped. `""` for a task-mode run; a `StopDecision` summary otherwise, so a
    # reader of the completion can answer "what did the run prove" without the ledger.
    goal_reason: str = ""

    @property
    def count(self) -> int:
        return len(self.completed)

    def to_payload(self) -> dict[str, object]:
        """The continue-as-new argument: plain dicts, so the payload shape is explicit in history."""
        return {
            "completed": [
                {
                    "task_id": summary.task_id,
                    "outcome": summary.outcome,
                    "cost_usd": summary.cost_usd,
                    "turn_seq": summary.turn_seq,
                    "approved_by": summary.approved_by,
                    "pr_url": summary.pr_url,
                    "verify_passed": summary.verify_passed,
                    "review_verdict": summary.review_verdict,
                }
                for summary in self.completed
            ],
            "spent_usd": self.spent_usd,
            "generations": self.generations,
            "outcome": self.outcome,
            "episodes": [episode.to_payload() for episode in self.episodes],
            "ci_state": self.ci_state,
            "ci_attempts": self.ci_attempts,
            "goal_reason": self.goal_reason,
        }

    @classmethod
    def from_payload(cls, payload: "dict[str, object] | None") -> "RunProgress":
        if not payload:
            return cls()
        from .goal import EpisodeRef as _EpisodeRef

        completed = tuple(
            TaskSummary(
                task_id=str(item["task_id"]),
                outcome=str(item["outcome"]),
                cost_usd=float(item.get("cost_usd") or 0.0),
                turn_seq=int(item.get("turn_seq") or 0),
                approved_by=item.get("approved_by"),  # type: ignore[arg-type]
                pr_url=item.get("pr_url"),  # type: ignore[arg-type]
                verify_passed=(
                    bool(item["verify_passed"])
                    if item.get("verify_passed") is not None
                    else None
                ),
                review_verdict=item.get("review_verdict"),  # type: ignore[arg-type]
            )
            for item in payload.get("completed") or ()  # type: ignore[union-attr]
        )
        return cls(
            completed=completed,
            spent_usd=float(payload.get("spent_usd") or 0.0),  # type: ignore[arg-type]
            generations=int(payload.get("generations") or 1),  # type: ignore[arg-type]
            outcome=payload.get("outcome") or "completed",  # type: ignore[arg-type]
            episodes=tuple(
                _EpisodeRef.from_payload(item) for item in payload.get("episodes") or ()  # type: ignore[union-attr]
            ),
            ci_state=payload.get("ci_state"),  # type: ignore[arg-type]
            ci_attempts=int(payload.get("ci_attempts") or 0),  # type: ignore[arg-type]
            goal_reason=str(payload.get("goal_reason") or ""),
        )


@dataclass(frozen=True)
class RunCompletion:
    run_id: str
    tasks: tuple[TaskSummary, ...]
    spent_usd: float
    generations: int
    outcome: RunOutcome
    # D5 goal mode: the stop condition's verdict, as `StopDecision.summary()`. Empty for a
    # task-mode run. Carried here (not only in the ledger) so "why did it stop" is answerable from
    # the workflow's own result.
    goal_reason: str = ""
    ci_attempts: int = 0
    # How many of ``tasks`` finished ``approved_unpublished``: the work is done, and publishing was
    # skipped because the repository had nothing to publish to (no GitHub remote). Additive and
    # derived from the summaries, never carried separately, so it cannot disagree with them.
    #
    # The RUN outcome stays ``completed`` in that case, and deliberately: at run level ``completed``
    # means "every task finished its work", which is true. Inventing a run-level
    # "completed_unpublished" would break every consumer of `RunOutcome` (the CLI's `--wait` finish
    # test, its exit code, the IDE's status projection, the soak harness) to say something the task
    # summaries already say precisely. This field is what lets a reader of the COMPLETION state it
    # without consulting them.
    unpublished_tasks: int = 0


@dataclass(frozen=True)
class TaskCompletion:
    """One task's result. `state` is the final reference set; nothing bulky rides along.

    Lives with the run contracts because that is who consumes it: `RunWorkflow` folds each child's
    completion into a `TaskSummary`, and a task's own caller reads the same fields. The `state` is a
    validated `TaskState` payload, so a caller that wants the details of what happened reads the same
    shape the activities produced.
    """

    run_id: str
    task_id: str
    outcome: str
    state: dict[str, Any]
    approved_by: str | None
    cost_usd: float
    turn_seq: int
    checkpoints: tuple[str, ...]
    engine: str
    # D5 goal mode: where this task's episode was written and what it hashed to, from the
    # `remember_episode` outcome the task itself produced. Carried HERE so the run does not call
    # `remember_episode` a second time to learn a path (which would append a second ledger note for
    # one episode). Both empty when no episode was recorded — a rejected plan or a stopped turn —
    # which is what makes "this task contributed no episode" visible rather than assumed.
    episode_path: str = ""
    episode_digest: str = ""

    def summary(self) -> "TaskSummary":
        """The small form a run keeps across a continue-as-new boundary."""
        pr = self.state.get("pr")
        verify = self.state.get("verify_ref")
        return TaskSummary(
            task_id=self.task_id,
            outcome=self.outcome,
            cost_usd=self.cost_usd,
            turn_seq=self.turn_seq,
            approved_by=self.approved_by,
            pr_url=pr.get("url") if isinstance(pr, dict) else None,
            # Goal-mode evidence, read from the state the task actually finished with.
            verify_passed=(
                bool(verify.get("passed")) if isinstance(verify, dict) and "passed" in verify else None
            ),
            review_verdict=(
                str(self.state["review_verdict"])
                if isinstance(self.state.get("review_verdict"), str)
                else None
            ),
        )


# The engines a run may select. Declared here as the literal tuple rather than imported, so this
# module stays a leaf of `contracts` (see the note above about the cycle).
ENGINES: Final[tuple[GraphEngine, ...]] = ("langgraph", "plain")
