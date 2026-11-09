"""Project rules and MCP on the ENGINE path (D5): the rule engine, fed to a durable turn.

Two different problems, with one shared cause: the durable engine path only ever sent a PROMPT. A
human running `codna fix` got their repository's conventions honoured because `fix_run` read the rules
and attached them as `project_guidance`; a durable run — the path that works unattended for hours —
did not, because nothing read them at all.

**Rules.** There is ONE reader, `codna.rules`, and this module does not re-implement any part of it
(discovery, scoping, precedence, budget). It decides WHERE the text goes and HOW MUCH of the prompt it
may occupy. The text goes into the session's ``injected_context``, which the engine folds into the
system prompt — verified in `run-server.bundle.mjs`:
``[args.prompt, args.injected_context].join(...)`` and ``if (args.injected_context) { ... }`` adds the
guidance section. A rules file that reached neither the prompt nor the system prompt would be a file
the agent never saw.

**Why the budget is a PARAMETER here, not a second truncation.** This module used to re-cut the
reader's output with ``[:MAX_RULES_CHARS]``. Two truncations in a row is two chances to lose a rule
silently, and the second one had no idea which rule it was cutting through. Now the prompt ceiling is
passed INTO the renderer, which drops whole rules and reports each one by name. The ceiling still
belongs to this module — "how big may a turn's prompt grow" is the engine path's question, not the
reader's — but it is now enforced once, where the accounting is.

**Why it is computed in the ACTIVITY, not the workflow.** The rules are a property of the WORKING
TREE, and a working tree changes while a run is in flight (a task's own turn may edit AGENTS.md). Two
consequences, both deliberate:

* The text must not enter workflow history. It is a file's contents — unbounded, repo-controlled, and
  exactly the class of data `TaskState` refuses for that reason. So the spec carries the POLICY
  (``attach_project_rules``), and the activity resolves the TEXT from the tree it is about to use.
* It must be read from the TREE THE TURN EDITS, not from the shared checkout: a task in its own
  worktree may be on a branch that changed the rules.

**MCP is carried, not silently omitted.** The engine's create-session contract accepts
``mcp_servers`` and turns each server into real, callable tools (`algenta/mcp-servers.ts`), so a
durable turn CAN be handed one. What travels in this spec is the configuration plus the NAMES of the
environment variables carrying its secrets — never a secret value, because a `TaskSpec` is workflow
history. `mcp_attachment_state` reports which of the three answers applies to a run (attached /
nothing configured / configured but disabled), and the run records it in the ledger. See
`cli/codna/durable/mcp.py` for the parsing and expansion rules; this module only owns the ANSWER, so a
run's ledger says truthfully what the engine path was given.

**The `extractPrompt` half of the plan's note is IDE-side, not here.** "`extractPrompt` sends only the
last user message" is about the Theia language-model bridge, which the sibling agent owns; the durable
path's equivalent — a task-mode plan prompt carrying no run history — is fixed by `goal.py`'s episode
recall, which appends the objective and the earlier tasks' episodes rather than a transcript.
"""
from __future__ import annotations

import sys
from typing import Final, Iterable

# The ATTACHMENT ANSWER lives beside the MCP config it describes. Re-exported from here because this
# module is where the engine path's capabilities are reported (rules + MCP are the two halves of "what
# besides a prompt does a durable turn get"), and moving a long-standing import would churn callers for
# no behavioural gain.
from .mcp import McpAttachment, mcp_attachment_state

__all__ = [
    "MAX_RULES_CHARS",
    "McpAttachment",
    "attach_project_rules",
    "mcp_attachment_state",
    "project_context_for",
    "project_rules_for",
]

# A hard ceiling on what a turn's system prompt may grow by. The reader's own default (16 KB) is larger
# than a prompt should carry: a 16 KB rules file is a larger system prompt on every turn of every task
# of every run, and the agent reads it every time. Bounded here rather than by trusting the reader,
# because "how big may a prompt get" is this module's question — but enforced by the renderer, which
# reports every rule it had to leave out instead of cutting a string in half.
MAX_RULES_CHARS: Final = 8_000


def project_rules_for(
    working_tree: str,
    *,
    enabled: bool = True,
    files_in_context: Iterable[str] | None = None,
    stream=None,
) -> str:
    """The repository's conventions for ``working_tree``, or ``""`` when there are none or disabled.

    Empty is a real answer, and deliberately not an error: a repository with no rules file is the
    normal case, and a run that failed because AGENTS.md was missing would be unusable on most repos.
    What must NOT happen is a read that silently yields nothing when the files ARE there — which is why
    the reader is the shared one, why ``enabled`` is an explicit parameter rather than a condition
    inside the caller, and why rules that do not fit are reported on ``stream`` (stderr by default).

    ``files_in_context`` is the set of repository-relative paths this turn is about. Supplying it is
    what makes a ``globs:``-scoped rule cost nothing on a turn it does not concern. Omitting it is the
    durable default and is NOT the same as passing ``[]``: an unknown file set keeps every glob rule
    attached (a durable agent may edit anything, so a globbed convention cannot be ruled out) and
    reports which ones matched nothing. Passing ``[]`` claims "nothing changed" and excludes them.
    """
    if not enabled:
        return ""
    from ..rules import guidance_for_repo

    guidance = guidance_for_repo(
        working_tree,
        files_in_context=files_in_context,
        max_bytes=MAX_RULES_CHARS,
        stream=stream if stream is not None else sys.stderr,
    )
    if not guidance or not guidance.strip():
        return ""
    return guidance.strip()


def attach_project_rules(working_tree: str, *, files_in_context: Iterable[str] | None = None):
    """The full structured answer for a turn: text plus what was kept, dropped and catalogued.

    A caller that wants to record WHY a turn carried the rules it did (a ledger entry, a diagnostic
    surface) uses this; ``project_rules_for`` is the string-only shortcut over the same engine call.
    """
    from ..rules import load

    return load(working_tree, files_in_context=files_in_context, max_bytes=MAX_RULES_CHARS)


def project_context_for(
    working_tree: str,
    *,
    enabled: bool = True,
    files_in_context: Iterable[str] | None = None,
    stream=None,
    include_user: bool = False,
) -> str:
    """The repository's rules AND its skills catalogue for ``working_tree``, within one budget.

    This supersedes :func:`project_rules_for` on the durable path now that a session may carry skills:
    the catalogue has to share the rules' ceiling, or the pair can each stay inside its own limit while
    together overflowing the prompt. ``MAX_RULES_CHARS`` remains the ONE number — it is the whole
    injected-context budget, not a per-skill extra.

    **``include_user`` defaults to ``False`` here, and that is a product decision, not a test
    convenience.** A durable run executes on a WORKER, possibly a shared one, and a `TaskSpec` is
    workflow history. Reading ``~/.cursor/skills`` or ``~/.codna/rules`` from that worker would put one
    machine's private conventions into a repository's turn — and would make two workers running the
    same recorded task produce two different prompts. A run's conventions are the repository's, which
    is why they are read from the tree the turn edits. The IDE path, which is the user's own machine,
    keeps the user layer ON.

    Empty is a real answer and is not an error: most repositories have no rules and no skills, and a run
    that failed because ``AGENTS.md`` was missing would be unusable there. What must NOT happen is a read
    that yields nothing when the files ARE present, which is why the reader is the shared one, why
    ``enabled`` is a parameter rather than a condition inside a caller, and why anything that did not fit
    is reported on ``stream`` (stderr by default).
    """
    if not enabled:
        return ""
    from ..rules import load_context

    loaded = load_context(
        working_tree,
        files_in_context=files_in_context,
        max_bytes=MAX_RULES_CHARS,
        include_user=include_user,
    )
    if stream is not None:
        for warning in loaded.warnings:
            print(f"codna: warning: [context] {warning.code}: {warning.source_path}: "
                  f"{warning.message}", file=stream)
        for entry in loaded.skill_catalogue.dropped:
            print(f"codna: warning: [context] {entry.name} ({entry.source_path}) did not fit the "
                  f"{MAX_RULES_CHARS}-byte context budget", file=stream)
    return loaded.text.strip()
