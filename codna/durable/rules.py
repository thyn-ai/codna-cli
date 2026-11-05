"""Project rules and MCP on the ENGINE path (D5): `.codna/rules` + AGENTS.md, read once per turn.

Two different problems, with one shared cause: the durable engine path only ever sent a PROMPT. A
human running `codna fix` got their repository's conventions honoured because `fix_run` read
`AGENTS.md`/`.codna/rules` and attached them as `project_guidance`; a durable run — the path that
works unattended for hours — did not, because nothing read them at all.

**Rules.** `project_rules.read_project_guidance` is the ONE reader (it already knows AGENTS.md,
`.codna/rules`, `.cursorrules`, `.cursor/BUGBOT.md` and the rest, and bounds the total). This module
does not re-implement it: it decides WHERE the text goes, which is the part that was missing. It goes
into the session's ``injected_context``, which the engine folds into the system prompt — verified in
`run-server.bundle.mjs`: ``[args.prompt, args.injected_context].join(...)`` and
``if (args.injected_context) { ... }`` adds the guidance section. A rules file that reached neither
the prompt nor the system prompt would be a file the agent never saw.

**Why it is computed in the ACTIVITY, not the workflow.** The rules are a property of the WORKING
TREE, and a working tree changes while a run is in flight (a task's own turn may edit AGENTS.md). Two
consequences, both deliberate:

* The text must not enter workflow history. It is a file's contents — unbounded, repo-controlled, and
  exactly the class of data `TaskState` refuses for that reason. So the spec carries the POLICY
  (``attach_project_rules``), and the activity resolves the TEXT from the tree it is about to use.
* It must be read from the TREE THE TURN EDITS, not from the shared checkout: a task in its own
  worktree may be on a branch that changed the rules.

**MCP is carried, not silently omitted.** The engine's create-session contract now accepts
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

from typing import Final

# The ATTACHMENT ANSWER lives beside the MCP config it describes. Re-exported from here because this
# module is where the engine path's capabilities are reported (rules + MCP are the two halves of "what
# besides a prompt does a durable turn get"), and moving a long-standing import would churn callers for
# no behavioural gain.
from .mcp import McpAttachment, mcp_attachment_state

__all__ = [
    "MAX_RULES_CHARS",
    "McpAttachment",
    "mcp_attachment_state",
    "project_rules_for",
]

# A hard ceiling on what a turn's system prompt may grow by. The reader already bounds itself
# (`project_rules._MAX_BYTES`), so this is a SECOND, smaller bound for the prompt's sake: a 16 KB
# rules file is a larger system prompt on every turn of every task of every run, and the agent reads
# it every time. Bounded here rather than by trusting the reader, because "how big may a prompt get"
# is this module's question and "how big may a rules file be" is the reader's.
MAX_RULES_CHARS: Final = 8_000


def project_rules_for(working_tree: str, *, enabled: bool = True) -> str:
    """The repository's conventions for ``working_tree``, or ``""`` when there are none or disabled.

    Empty is a real answer, and deliberately not an error: a repository with no rules file is the
    normal case, and a run that failed because AGENTS.md was missing would be unusable on most repos.
    What must NOT happen is a read that silently yields nothing when the files ARE there — which is
    why the reader is the shared one, and why ``enabled`` is an explicit parameter rather than a
    condition inside the caller.
    """
    if not enabled:
        return ""
    from ..project_rules import read_project_guidance

    guidance = read_project_guidance(working_tree)
    if not guidance or not guidance.strip():
        return ""
    return guidance.strip()[:MAX_RULES_CHARS]

