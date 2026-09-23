"""Make agent-produced unified diffs applicable before they reach `git apply`.

Model output is not a byte-exact diff: it arrives wrapped in a markdown fence, with CRLF line
endings, with the single leading space of blank context lines stripped by a trailing-whitespace
pass ("error: corrupt patch at line N"), or with hunk headers whose line counts are off by a few.
The first live `@codna fix` to reach this step (thyn-ai/algenta#1003, 2026-09-17) died on exactly
such a patch. None of these are the patch being wrong about the code, so repair them here and let
`git apply --recount` absorb the count drift; a patch that still fails is rejected with git's own
first error line.
"""
from __future__ import annotations

import re

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_FENCE = re.compile(r"^```[\w+.-]*\s*$")
_STRUCTURAL = ("diff --git ", "index ", "--- ", "+++ ", "similarity index", "rename from", "rename to",
               "new file mode", "deleted file mode", "old mode", "new mode", "Binary files")

# Tried in order by every apply site; the first set whose `--check` passes is used for the real apply.
GIT_APPLY_FLAG_SETS: tuple[tuple[str, ...], ...] = ((), ("--recount",), ("--recount", "--ignore-whitespace"))


def normalize_unified_diff(text: str) -> str:
    """Return *text* as a diff `git apply` can read; a diff that needs nothing comes back unchanged
    except for a guaranteed single trailing newline."""
    if not text or not text.strip():
        return text or ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _FENCE.match(lines[0]):
        lines.pop(0)
    if lines and _FENCE.match(lines[-1]):
        lines.pop()
    out: list[str] = []
    in_hunk = False
    for line in lines:
        if _HUNK.match(line):
            in_hunk = True
        elif line.startswith(_STRUCTURAL):
            in_hunk = False
        elif in_hunk and line == "":
            line = " "  # a context line whose leading space was stripped
        out.append(line)
    return "\n".join(out) + "\n"
