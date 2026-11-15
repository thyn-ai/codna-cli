"""The PROMPT builders: what a turn is asked, as pure functions of its inputs (D5).

Split out of `contracts.py` at the repository's 1000-line ceiling (`test_modularity.py`), on a real
boundary: this module answers "what TEXT does the model receive", and `contracts.py` answers "what
SHAPE crosses the wire". A reviewer changing a prompt wants to read the prompt, not the schema.

Every function here is PURE and APPENDS rather than interleaves, which is what makes a recorded
prompt replay byte for byte:

* ``plan_prompt`` appends human steering, then the goal block (D5: the run's objective plus the
  episodes of earlier tasks). A task-mode run passes no goal block, so its prompt is byte-identical to
  what it always was — the property the replay cassettes depend on.
* ``implement_prompt`` appends steering only: the goal is a PLANNING concern, and repeating it on
  every implement turn would grow the prompt without changing what the turn does.
* ``pr_title`` is derived from the brief's first line, bounded, so a title cannot become a paragraph.

Re-exported from `contracts`, so every caller keeps importing one module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .contracts import TaskSpec

__all__ = ["implement_prompt", "plan_prompt", "pr_title"]


def plan_prompt(brief: str, steer: Sequence[str] = (), goal_context: str = "") -> str:
    """The read-only planning prompt: a pure function of its inputs, so a retry plans the same.

    Steering notes are appended rather than woven in, so the order is preserved and the text stays
    deterministic — which is what makes a recorded history replayable. ``goal_context`` (D5 goal mode)
    is the run's objective plus the episodes of earlier tasks, and it is appended LAST so a task-mode
    prompt (empty goal context) is byte-identical to what it always was.

    **THE TWO OUTCOMES, stated rather than assumed.** This turn used to end with "Answer with an
    ordered list of steps, each naming the files it touches" and nothing else, so the only shape the
    model was ever offered was a PLAN. A message that is not a task therefore came back as a
    plan-shaped refusal — measured (`run-20261115T194227Z`, brief ``hi``): *"The task is a greeting
    ("hi") with no actionable change… Plan: No-op — no changes required… Files touched: none."* The
    user asked "Hi" and was handed a work plan for doing nothing.

    The prompt does not try to CLASSIFY the brief (a haystack of heuristics is a classifier nobody
    can test). It states both legitimate outcomes and lets the turn answer with the one that fits,
    which is a rule that generalises to any non-task message: a greeting, a question, a request for
    an explanation. "Do not invent work" stays explicit, because the failure on the other side —
    fabricating a change to satisfy the format — is the one the no-hallucination policy forbids.
    """
    lines = [
        "Produce a plan for this task. Do NOT modify any file: this turn is read-only.",
        "",
        "Task:",
        brief.strip(),
        "",
        # The task-branch instruction is unchanged, so a real brief is planned exactly as before.
        "If the task asks for a change, answer with an ordered list of steps, each naming the files "
        "it touches.",
        # …and this is the branch a non-task message takes, instead of the no-op plan it used to get.
        "If the task asks for NO change — a greeting, a question, or a request you can answer from "
        "the conversation and your own knowledge — answer it directly and briefly, saying plainly "
        "that no change is needed. Do not produce a plan for work that was not requested, and do not "
        "invent work to fill one.",
    ]
    if steer:
        lines.extend(["", "Human steering (oldest first):"])
        lines.extend(f"- {note.strip()}" for note in steer)
    if goal_context.strip():
        lines.append(goal_context.rstrip())
    return "\n".join(lines)


def implement_prompt(brief: str, steer: Sequence[str] = ()) -> str:
    """The implement prompt: the brief, plus whatever the human steered since the last turn.

    Same two outcomes as `plan_prompt`: a brief that asks for a change is implemented, and one that
    asks for nothing is ANSWERED. Without this branch the implement turn could only answer a greeting
    by describing the empty diff it did not write ("a no-op completed… 0 files touched"), which is the
    same plan-shaped refusal one step later. Nothing is written either way — the no-op is the honest
    outcome, and saying it in one sentence is the difference between an answer and a work report.
    """
    lines = [
        "Implement this task in the repository. Keep the diff focused and the tests green.",
        "",
        "Task:",
        brief.strip(),
        "",
        "If the task asks for NO change — a greeting, a question, or a request you can answer from "
        "the conversation and your own knowledge — answer it directly and briefly, saying plainly "
        "that nothing needed to change. Do not invent a change, and do not write a work report about "
        "a diff you did not make.",
    ]
    if steer:
        lines.extend(["", "Human steering (oldest first):"])
        lines.extend(f"- {note.strip()}" for note in steer)
    return "\n".join(lines)


def pr_title(spec: "TaskSpec") -> str:
    """The pull request title: the brief's first line, bounded. Deterministic in the spec.

    A title derived from a MODEL's summary would be free text that reads like a sentence ("codna: fix
    Here's a summary of every change made") and would differ between attempts of the same task. The
    brief's own first line is what the human wrote, and it is the same on a retry.
    """
    first = spec.brief.strip().splitlines()[0].strip() if spec.brief.strip() else ""
    title = first or f"codna: {spec.task_id}"
    return title[:256]
