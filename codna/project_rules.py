"""Project guidance (AGENTS.md / .codna/rules / .cursorrules) surfaced to codna's fix agent.

`codna init` scaffolds AGENTS.md; this reads it (+ common alternatives) and the fix/review flow attaches
it to the engine signal as ``project_guidance`` so the agent can honor project conventions (test command,
style, forbidden paths). The CLI surfaces it here; the decision-engine consuming it to actually steer the
agent is the paired follow-up. Bounded in size so a large rules file can't bloat the request.
"""
from __future__ import annotations

import os

# Checked in order; all present files are concatenated (AGENTS.md is the primary, emerging standard).
# `.cursor/BUGBOT.md` / `BUGBOT.md` are Cursor Bugbot's review-rules files — reading them gives repos
# already configured for Bugbot drop-in guidance under Codna with zero migration.
_RULES_FILES = ("AGENTS.md", ".codna/rules.md", ".codna/rules", ".codna/review.md",
                ".cursorrules", ".cursor/rules.md", ".cursor/BUGBOT.md", "BUGBOT.md")
_MAX_BYTES = 16 * 1024


def read_project_guidance(repo_dir: str | None) -> str | None:
    """Return concatenated project guidance from the repo root, or None if there is none."""
    root = os.path.abspath(os.path.expanduser(repo_dir or "."))
    if not os.path.isdir(root):
        return None
    parts: list[str] = []
    for rel in _RULES_FILES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read().strip()
        except OSError:
            continue
        if text:
            parts.append(f"# {rel}\n{text}")
    if not parts:
        return None
    return "\n\n".join(parts)[:_MAX_BYTES]
