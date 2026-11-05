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
    """
    lines = [
        "Produce a plan for this task. Do NOT modify any file: this turn is read-only.",
        "",
        "Task:",
        brief.strip(),
        "",
        "Answer with an ordered list of steps, each naming the files it touches.",
    ]
    if steer:
        lines.extend(["", "Human steering (oldest first):"])
        lines.extend(f"- {note.strip()}" for note in steer)
    if goal_context.strip():
        lines.append(goal_context.rstrip())
    return "\n".join(lines)


def implement_prompt(brief: str, steer: Sequence[str] = ()) -> str:
    """The implement prompt: the brief, plus whatever the human steered since the last turn."""
    lines = [
        "Implement this task in the repository. Keep the diff focused and the tests green.",
        "",
        "Task:",
        brief.strip(),
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
