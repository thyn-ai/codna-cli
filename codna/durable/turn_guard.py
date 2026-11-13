"""Fail a turn FAST when it is repeating one deterministic tool failure.

── THE DEFECT THIS CLOSES, MEASURED ON THE SECOND APP A REAL USER BUILT ────────────────────────

Run `run-20261112T232923Z` (a small calculator app, started from the packaged app's New task
sheet) ended with:

    seq 3131  session.turn_failed  {"error": "Error: turn exceeded timeout budget of 600000ms",
                                    "turn_id": "turn_session_..._2"}

so `verify` and `review` never ran and the run's outcome was `failed`. The budget was NOT the
cause and raising it would not have helped. The implement turn's own ledger says exactly what it
spent the ten minutes on — 11 iterations, 10 tool calls, EVERY ONE of them the same `editor`
failure:

    seq  527  editor  {"new_text": "<!DOCTYPE html>…", "path": "index.html"}
              → 529  {"error": "Editor operation failed: Parameter `old_text` is required when
                      editing an existing file without `insert_line`", "success": false}
    seq  569  the same call, the same error
    … then the SAME error for styles.css, app.js, tests/calc.test.js and back again
       (625, 661, 743, 797, 861, 937, 1027, 1123) → 10 identical failures

Each iteration costs one model round-trip (~30–60 s against a local model), so ten of them is the
whole budget. A turn that repeats a DETERMINISTIC failure cannot succeed on the next iteration:
nothing about the tree changed in between. Waiting for the deadline only delays the same failure
by nine minutes and destroys the diagnosis, because `turn exceeded timeout budget` says nothing
about the editor parameter that was actually wrong.

── WHY THIS IS NOT "RAISE THE BUDGET" ───────────────────────────────────────────────────────────

The control is the FIRST app, which completed: `run-20261112T230703Z`'s implement turn ran 11
iterations and 9 tool calls in 54 s (turn.started seq 274 → turn.finished seq 888) and verified
clean. A legitimate small-app implement turn is ~1/11th of the 600 s budget, so the budget is not
too small for the work — one turn in this population was pathological. `DEFAULT_TURN_BUDGET_MS`
stays exactly where it is (agent-core/vendor/cline/algenta/provider-timeout.ts): raising a timeout
to outlast a loop hides the loop and bills for it.

── WHAT TRIPS, AND WHAT DOES NOT ───────────────────────────────────────────────────────────────

The signal is a repeated deterministic failure of ONE CALL — same tool, same error — counted per
(tool, error) across the whole turn, and cleared the moment that tool SUCCEEDS. Each part closes a
hole that was measured, not imagined:

  * **Counted PER (tool, error), not in one slot.** The first version kept a single "last failure"
    slot, which any different failure reset — so `A, B, A, B, …` (an `editor` call failing, a
    policy-refused `graph_query`, the `editor` call again) never tripped however long it ran. A
    runaway that alternates between two bad calls is still a runaway.
  * **The error is CANONICALISED before comparing.** A deterministic tool error routinely embeds a
    volatile tail — `EEXIST … mkdir '/tmp/x/tests'`, a request id, a timestamp. Comparing raw text
    let one failing call escape detection indefinitely by changing only that suffix. UUIDs, ISO
    timestamps, hex ids and absolute paths are collapsed, so `EEXIST … mkdir '<path>'` is one error
    however many directories it names. BARE DIGITS ARE NOT collapsed: a changing count (`line 12` vs
    `line 48`) is evidence of progress.
  * **…EXCEPT that a MISSING-SOURCE error keeps its path, and that exception is measured.** For
    `ENOENT`/`path is not a file` the path is the ARGUMENT of the call, not a volatile detail: a plan
    turn exploring a fresh tree reads several files that do not exist yet, one per path. Collapsing
    those merged them into a single key and killed the turn — measured on `run-20261113T051836Z`,
    seven reads of seven DIFFERENT missing files tripped the guard on a read-only PLAN turn before a
    single file was written. Keeping the path there drops that turn to 2 while every true runaway in
    the corpus still trips at 7-10, because a real `ENOENT` loop repeats the SAME path.
    The exception is NARROW ON PURPOSE: `ENOTDIR`/`not a directory` are absence errors too, and for
    `editor` the path in one is the OUTPUT the call chose, so treating them as arguments re-opened a
    hole where the same editor failure against different outputs never tripped. Only the two literals
    that report a missing SOURCE are exempt; everything else collapses its path.
  * **Arguments are deliberately NOT part of the identity.** Tempting, and wrong for THIS defect:
    the measured runaway failed the identical error against `index.html`, `styles.css`, `app.js` and
    `tests/calc.test.js` in turn. Keying on arguments made it invisible. What separates "stuck" from
    "iterating" is PROGRESS, which is why the count is reset by a success of that tool and not by a
    change of arguments.
  * **A SUCCESS of that tool clears its count.** "fail, succeed, fail" is an ordinary edit
    sequence (a read-then-edit dance), not a loop. Other tools' counts are untouched, which is what
    lets an interleaved loop still be seen.
  * **A POLICY refusal is not counted.** `Tool "graph_query" is disabled by policy`, or the
    engine's `… rejected before execution: Model tried to call unavailable tool 'x'`, is
    deterministic too, but it says the TOOLSET is wrong rather than that the call is — and the
    todo run recovered from exactly one of them by choosing another tool. Both literals are in the
    corpus; matching only the first left the second class counted as a loop.

── THE THRESHOLD, SET FROM THE CORPUS RATHER THAN FROM INTUITION ───────────────────────────────

Three was the original guess and it is measurably too aggressive. Replaying this detector over the
51 run ledgers on this machine (51 turns: 38 completed, 8 failed) at three identical failures
KILLED THREE turns that went on to COMPLETE. The clearest is `run-20261112T223954Z`'s implement
turn: it failed the same `editor` call three times (`old_text` required), then READ the file,
learned the argument it was missing, and edited it successfully at seq 1315. A threshold of three
would have aborted the turn before the read that fixed it. So three identical failures do NOT prove
the next call is the same call — a real model recovers from them.

At six the corpus still contains one completed turn (a turn that surrendered after six), so the
smallest value with NO false positive over all 51 real turns is SEVEN. It also still stops a
runaway while there is budget left: this file's motivating run repeated its call TEN times and the
other measured runaway (`run-20261112T231653Z`) EIGHT times, so both are stopped with iterations
to spare. Erring late is the correct direction — a false kill destroys work that was about to
succeed, while a late kill only forfeits iterations, and the turn's own deadline remains the
backstop for a runaway that never repeats one error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final, Mapping

__all__ = [
    "MAX_IDENTICAL_TOOL_FAILURES",
    "TOOL_CALL_EVENT",
    "TOOL_LOOP_EVENT",
    "TurnGuard",
    "ToolFailureLoop",
]

#: The stream event carrying one finished tool call (engine `execution.ts` → `artifacts.ts`).
TOOL_CALL_EVENT: Final = "agent.content_end"

#: The ledger note recording this verdict, so a RETRY reports the loop rather than the abort it caused.
#: An activity retry re-reads the ledger (`turn_driver._scan_ledger`), and without this it would
#: re-attach to the turn we aborted and report `activity_turn_failed` — replacing the real cause.
TOOL_LOOP_EVENT: Final = "turn.tool_loop"

#: Repeated identical failures of one call that prove the next iteration is the same call.
#: SEVEN because that is the smallest value with no false positive over the measured run corpus
#: (51 ledgers, 51 turns: 38 completed, 8 failed) — see the module header for the turns it would
#: otherwise have killed.
MAX_IDENTICAL_TOOL_FAILURES: Final = 7

#: A refusal the engine writes when the TOOLSET does not contain the tool, or policy forbids it.
#: Deterministic, but it says the tool is unavailable rather than that the call was wrong, and a
#: model may recover by choosing another tool — so it is not a loop. Both literals are measured.
_POLICY_REFUSAL_MARKERS: Final = (
    "disabled by policy",
    "rejected before execution",
    "unavailable tool",
)

#: Volatile tokens inside a deterministic error message. Collapsed so that a changing suffix cannot
#: make one failing call look like a series of new ones.
#:
#: BARE DIGITS ARE DELIBERATELY NOT COLLAPSED. They read as volatile, and collapsing them is how the
#: first draft merged genuinely different errors (`failure 0` … `failure 6`, or `Error at line 12`
#: versus `line 48`) into one, which trips the guard on a model that is iterating. A count that
#: changes between messages is evidence of progress, not noise.
_UUID_RE: Final = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_ISO_TS_RE: Final = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
_HEX_RE: Final = re.compile(r"\b[0-9a-f]{8,}\b")
_PATH_RE: Final = re.compile(r"(?:/[\w.\-]+){2,}/?")

#: Errors whose SUBJECT IS THE PATH — the file or directory was not where the call looked.
#:
#: This class is canonicalised WITHOUT collapsing the path, and the difference is measured rather
#: than stylistic. `read_files` on a repository that does not have the file yet reports
#: `ENOENT … stat '<path>'`, so a plan turn exploring a fresh tree produces a run of these with a
#: DIFFERENT path in each — `package.json`, then `index.html`, then `tests/counter.test.js`. Under
#: whole-error path collapsing those merged into one key: measured on `run-20261113T051836Z`, seven
#: such reads of seven DIFFERENT missing files tripped the guard and killed a PLAN turn before it
#: wrote anything. That is the false positive this class exists to remove: for an absence error the
#: path IS the argument of the call, so two paths are two calls, not one call repeated.
#:
#: A genuine `ENOENT` runaway is still caught, because it repeats the SAME path — the key stays
#: identical and the count still reaches the threshold. Replaying the corpus with this rule, every
#: true runaway still trips at 7-10 while that plan turn drops to 2.
#:
#: `ENOTDIR` / `not a directory` ARE DELIBERATELY ABSENT, though they are absence errors too.
#: Measured, not assumed: the corpus contains an `editor` failure of exactly that shape
#: (`ENOTDIR: not a directory, open '<path>'`, `run-20261112T222825Z` seq 3115), and for `editor` the
#: path is the OUTPUT the call chose rather than the thing it was looking for. Keeping those arms
#: meant the same `editor` failure against several output paths produced several canonical keys, so a
#: deterministic editor loop never tripped — the review reproduced it with
#: `Cannot write: target path is not a directory, /repo/outputs/a` and `…/b` yielding two keys.
#: Restricting the class to the two literals `read_files`/`editor` actually use to say a SOURCE was
#: missing closes that hole without slowing any real runaway: replayed over every ledger on this
#: machine the narrowed rule trips on every runaway the wider one did, at the same seq.
_ABSENCE_RE: Final = re.compile(
    r"ENOENT|no such file or directory|path is not a file",
    re.IGNORECASE,
)

#: The keys a tool result's error text may live under, in the order the engine shapes put it.
_OUTPUT_TEXT_KEYS: Final = ("error", "result", "content")


@dataclass(frozen=True)
class ToolFailureLoop:
    """Why the guard tripped, as the fact a failure report quotes. Never a guess."""

    tool_name: str
    error: str
    count: int

    def as_details(self) -> dict[str, Any]:
        return {"tool": self.tool_name, "error": self.error, "count": self.count}


def _canonical_error(error: str) -> str:
    """The comparison form of one error: whitespace collapsed, volatile tokens replaced.

    THE PATH IS KEPT for an ABSENCE error, because there the path is what the call asked for rather
    than a volatile detail of the failure — see `_ABSENCE_RE` for the plan turn that was killed by
    collapsing it. For every other error the path is incidental (`EEXIST … mkdir '<dir>'` names a
    directory the call did not choose), so it is collapsed and cannot be used to escape detection.
    """
    text = " ".join(error.split())
    text = _UUID_RE.sub("<uuid>", text)
    text = _ISO_TS_RE.sub("<ts>", text)
    text = _HEX_RE.sub("<hex>", text)
    if _ABSENCE_RE.search(text):
        return text
    return _PATH_RE.sub("<path>", text)


def _result_error(value: Any) -> str | None:
    """The error text one tool RESULT reported, or ``None`` when it reported no failure.

    Every shape the engine actually emits is read, because each one that was missed made the guard
    blind to a whole class of loops. MEASURED shapes in the run corpus:

    * the `editor` tool: ``{"error": …, "query": …, "result": "", "success": false}``;
    * `read_files` / `search_codebase`: a LIST of such dicts — a batch where every entry failed is
      a failed call, while one success in the batch is progress;
    * MCP tools (`browser__*`): ``{"isError": true, "content": [{"type": "text", "text": …}]}``,
      which carries NEITHER `success` NOR a string `error`;
    * a bare string output.

    Total: anything unrecognisable is "no failure reported" rather than a failure. Failing a run on
    an event nobody understood would be worse than missing a loop.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        items = [item for item in value if isinstance(item, Mapping)]
        if not items:
            return None
        # Any success in a batch is progress: the call did part of what it was asked.
        if any(item.get("success") is True for item in items):
            return None
        errors = [
            str(item["error"]).strip()
            for item in items
            if isinstance(item.get("error"), str) and item["error"].strip()
        ]
        if errors:
            return "; ".join(errors)
        # A batch of plain MCP content parts, e.g. `[{type: text, text: …}]`, carries its whole
        # message in `text` and has no `error` key at all.
        texts = [
            str(item.get("text")).strip()
            for item in items
            if isinstance(item.get("text"), str) and item["text"].strip()
        ]
        return "; ".join(texts) if texts else None
    if isinstance(value, Mapping):
        if value.get("success") is True:
            return None
        if value.get("isError") is True:
            nested = _result_error(value.get("content"))
            if nested is not None:
                return nested
            for key in (*_OUTPUT_TEXT_KEYS, "message"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            # The flag alone is the only evidence there is; it is still a failure.
            return "tool reported isError"
        for key in _OUTPUT_TEXT_KEYS:
            nested = _result_error(value.get(key))
            if nested is not None:
                return nested
        return None
    return None


def _tool_result_error(payload: Mapping[str, Any]) -> str | None:
    """The error text one finished tool call reported, or ``None`` when it did not fail."""
    if payload.get("contentType") != "tool" or payload.get("success") is True:
        return None
    return _result_error(payload.get("output"))


def _is_policy_refusal(error: str) -> bool:
    """True for a toolset refusal — deterministic, but not evidence of a repeated bad CALL."""
    return any(marker in error for marker in _POLICY_REFUSAL_MARKERS)


@dataclass
class TurnGuard:
    """Counts repeated identical tool failures for ONE turn and reports the loop.

    Stateful by design and owned by the streaming driver, which constructs one per turn: the
    ledger is per turn, so the count must be too. Pure with respect to everything else — the same
    events in the same order always produce the same verdict.
    """

    max_identical_failures: int = MAX_IDENTICAL_TOOL_FAILURES
    #: (tool, canonical error) -> how many times that call has failed since the tool last succeeded.
    _failures: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.max_identical_failures, int) or self.max_identical_failures < 1:
            raise ValueError("max_identical_failures must be a positive integer")

    def observe(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> ToolFailureLoop | None:
        """Fold one stream event in. Returns the loop exactly when it trips, else ``None``.

        A SUCCESS of a tool clears that tool's counts: the tree changed, so the next failure is new
        information rather than a repeat of a call made against the old tree. Other tools' counts
        are untouched, which is what lets an interleaved loop still be seen.
        """
        if event_type != TOOL_CALL_EVENT or payload.get("contentType") != "tool":
            return None
        tool_name = payload.get("toolName")
        if not isinstance(tool_name, str) or not tool_name:
            # An unattributed result cannot be compared to anything, so it neither extends nor
            # clears a count. Silence is the honest answer: this is not evidence either way.
            return None
        error = _tool_result_error(payload)
        if error is None:
            self._clear_tool(tool_name)
            return None
        if _is_policy_refusal(error):
            # The toolset is wrong, not the arguments. Not counted, and not treated as a success
            # either: it says nothing about the state of the tree.
            return None
        key = (tool_name, _canonical_error(error))
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count < self.max_identical_failures:
            return None
        return ToolFailureLoop(tool_name=tool_name, error=error, count=count)

    def _clear_tool(self, tool_name: str) -> None:
        """Drop every count for one tool, after that tool succeeded."""
        for key in [key for key in self._failures if key[0] == tool_name]:
            del self._failures[key]
