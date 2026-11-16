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

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .contract_primitives import ContractError
from .review_feedback import review_feedback_from_payload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .contracts import TaskSpec
    from .review_feedback import ReviewFeedback

__all__ = ["implement_prompt", "implement_prompt_for_state", "plan_prompt", "pr_title"]


def implement_prompt_for_state(
    state: Mapping[str, Any], brief: str, steer: Sequence[str] = ()
) -> str:
    """The implement prompt for a state PAYLOAD — the ONE place a retry's findings are read.

    Both engines build their implement prompt through this, so a retry cannot be told about the
    rejection on one engine and left blind on the other. Reading it from the payload (rather than
    threading a new parameter through `Step`/`_step`) also means the feedback travels exactly the way
    every other fact a turn depends on does: in the state, which is what a replay rebuilds.

    ``fail=ContractError`` makes a malformed record a NAMED contract failure at the workflow boundary
    rather than a prompt rendered from a half-decoded dict.
    """
    feedback = review_feedback_from_payload(state.get("review_feedback"), fail=ContractError)
    return implement_prompt(brief, steer, feedback)


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


def implement_prompt(
    brief: str,
    steer: Sequence[str] = (),
    review_feedback: "ReviewFeedback | None" = None,
) -> str:
    """The implement prompt: the brief, whatever the human steered, and why a RETRY was rejected.

    Same two outcomes as `plan_prompt`: a brief that asks for a change is implemented, and one that
    asks for nothing is ANSWERED. Without this branch the implement turn could only answer a greeting
    by describing the empty diff it did not write ("a no-op completed… 0 files touched"), which is the
    same plan-shaped refusal one step later. Nothing is written either way — the no-op is the honest
    outcome, and saying it in one sentence is the difference between an answer and a work report.

    ``review_feedback`` is the third input, and it closes a loop that could not converge without it:
    `after_review` sends a rejected change straight back to `implement`, so a run whose review says
    "changes requested" was re-asked to do the SAME task with NO knowledge of what was wrong. Measured
    (`run-20261116T005743Z`, build `6f7a904d`): three `changes_requested` verdicts on the unchanged SHA
    `8cce0fa1`, and the 3rd and 4th implement prompts in the ledger were BYTE-IDENTICAL.

    APPENDED, never interleaved, for the reason the module docstring gives: a task-mode run passes
    nothing here, so every prompt recorded before this parameter existed is still produced byte for
    byte. The block is bounded by the RECORD (`review_feedback.MAX_FINDINGS_RECORDED`), and the count
    it could not carry is stated rather than dropped.
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
    if review_feedback is not None:
        lines.extend(_review_feedback_block(review_feedback))
    return "\n".join(lines)


def _review_feedback_block(feedback: "ReviewFeedback") -> list[str]:
    """The 'the reviewer rejected your previous attempt' block, as prompt lines.

    WHY IT IS RENDERED THIS WAY. The model is being asked to change work it has already done once, so
    the block has to answer three questions in the order it would be asked: *was this a rejection*
    (it was), *on what attempt* (the SHA, so a model that already changed the tree can tell which
    state the reasons are about), and *what specifically to fix* (one line per finding, most severe
    first, with a path:line and an actionable statement).

    Deterministic in the record alone: no clock, no I/O, no dict iteration, and the record's findings
    are already in a total order (`review_feedback._sort_key`), so a replay renders the same text.

    The omission line is NOT decoration. A truncated list that does not say it is truncated reads as
    the complete set of objections, which would send the retry to fix a subset and believe it was done.
    """
    total = len(feedback.findings) + feedback.omitted
    lines = [
        "",
        _REVIEW_FEEDBACK_HEADING,
        f"Your previous attempt (commit {feedback.sha[:12]}) was REVIEWED and REJECTED. This is a "
        f"retry: the findings below are the reviewer's reasons ({len(feedback.findings)} of {total} "
        "shown, most severe first). Address every one of them in the work you do now.",
    ]
    for index, finding in enumerate(feedback.findings, start=1):
        location = f"{finding.path}:{finding.line}"
        category = f"/{finding.category}" if finding.category else ""
        lines.append("")
        lines.append(f"{index}. [{finding.severity}{category}] {location} — {finding.title}")
        if finding.explanation:
            lines.append(f"   {finding.explanation}")
    if feedback.omitted:
        lines.append("")
        lines.append(
            f"({feedback.omitted} further finding(s) were omitted from this list for length. They are "
            "recorded in the run's review ledger entry for that commit; fix what is shown here, and "
            "treat the diff as still needing a careful read.)"
        )
    return lines


#: The heading that marks the block. A constant because a test and a reader both need to find it, and
#: because it is the one string that says "this is a retry after a rejection" out loud.
_REVIEW_FEEDBACK_HEADING: str = (
    "The reviewer rejected your previous attempt for these reasons "
    "(from the review of the last commit; fix them, do not repeat them):"
)


def pr_title(spec: "TaskSpec") -> str:
    """The pull request title: the brief's first line, bounded. Deterministic in the spec.

    A title derived from a MODEL's summary would be free text that reads like a sentence ("codna: fix
    Here's a summary of every change made") and would differ between attempts of the same task. The
    brief's own first line is what the human wrote, and it is the same on a retry.
    """
    first = spec.brief.strip().splitlines()[0].strip() if spec.brief.strip() else ""
    title = first or f"codna: {spec.task_id}"
    return title[:256]
