"""Project guidance (AGENTS.md / .codna/rules / .cursorrules) surfaced to codna's fix agent.

**This module is now a compatibility shim.** It used to BE the reader: eight fixed root filenames
concatenated in order, then sliced with ``[:MAX_BYTES]`` — which silently cut the tail of whichever
file happened to sort last. That is the defect the rule engine (`codna.rules`) removes, and the reason
every consumer below now shares one reader instead of disagreeing with the vendored SDK's.

What is preserved here, deliberately and with tests:

  * ``read_project_guidance(repo_dir) -> str | None`` — same signature, same ``None``-when-absent
    contract, same human-readable ``# <rel path>`` block labels. Existing callers (``fix_run``,
    ``review``, ``packaged_agent_runner``) are unchanged.
  * ``_MAX_BYTES`` — still the default budget, so the size ceiling a caller sees today is the ceiling
    it gets. The difference is that exceeding it now REPORTS what did not fit instead of truncating.

What is new, and only reachable through this module because consumers asked for it:

  * ``guidance_for_files`` — the same read, scoped to the files a turn is about. A rule whose
    ``globs`` do not match anything in context is not paid for. This is what lets IDE chat and
    ``codna fix`` agree: both call the engine, and the engine's answer depends on the same inputs.
  * ``read_rules`` — the structured result (kept / dropped / catalogued), for a caller that wants to
    report rather than only to prompt.

Nothing here re-implements discovery, matching or rendering. If it did, the two-readers problem would
come back in a new coat.
"""
from __future__ import annotations

from typing import Iterable

from . import rules as _rules

# Checked in order; all present files are concatenated (AGENTS.md is the primary, emerging standard).
# `.cursor/BUGBOT.md` / `BUGBOT.md` are Cursor Bugbot's review-rules files — reading them gives repos
# already configured for Bugbot drop-in guidance under Codna with zero migration.
#
# NOTE: this tuple is now DOCUMENTATION of what the engine reads at the repository root, not the
# literal read order. Discovery is the single source of truth (`codna.rules.discovery._LEGACY_ROOT_FILES`
# and the `.codna/rules`/`.cursor/rules` package walk); keeping a duplicate list here would be the very
# drift this refactor removed. It is retained because callers and docs reference it, and a test pins it
# to the engine's legacy set so the two cannot silently diverge.
_RULES_FILES = ("AGENTS.md", ".codna/rules.md", ".codna/rules", ".codna/review.md",
                ".cursorrules", ".cursor/rules.md", ".cursor/BUGBOT.md", "BUGBOT.md")
_MAX_BYTES = _rules.DEFAULT_MAX_BYTES


def read_project_guidance(repo_dir: str | None) -> str | None:
    """Return this repository's guidance as prompt text, or None if there is none.

    Kept byte-compatible in SPIRIT with the historical reader: a repository with nothing to say gets
    ``None``, and every rendered block is labelled with the path it came from. The change is that the
    strings are produced by the engine, so scoping, precedence and the budget all apply — and a rule
    that does not fit is reported on stderr rather than lost.
    """
    return _rules.guidance_for_repo(repo_dir, max_bytes=_MAX_BYTES)


def read_rules(
    repo_dir: str | None,
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = _MAX_BYTES,
) -> _rules.LoadedRules:
    """The structured result: what was read, what applied, what was kept and what was dropped."""
    return _rules.load(
        repo_dir,
        files_in_context=files_in_context,
        explicit_mentions=explicit_mentions,
        max_bytes=max_bytes,
    )


def guidance_for_files(
    repo_dir: str | None,
    files_in_context: Iterable[str] | None = None,
    *,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = _MAX_BYTES,
) -> str | None:
    """Guidance scoped to ``files_in_context`` — the entry point a turn with known target files uses.

    ``None`` means "the caller does not know which files this turn touches" (a durable run, whose agent
    may edit anything). Every glob rule is then attached, because it cannot be ruled out, and the ones
    that matched nothing are reported. Passing a LIST — ``[]`` included — is a claim about the file set,
    and a glob rule matching none of it is correctly excluded. That distinction is the difference
    between scoping and silently under-applying.
    """
    return _rules.guidance_for_repo(
        repo_dir,
        files_in_context=files_in_context,
        explicit_mentions=explicit_mentions,
        max_bytes=max_bytes,
    )
