"""The Codna rule engine: one reader, many consumers, four modes, deterministic rendering.

**Why this is a package and not one file.** ``test_modularity.py`` caps every module at 1000 lines,
and the rule engine genuinely has four separable concerns that each deserve their own tests: parsing
(``frontmatter``), matching (``globs``), finding (``discovery``), and deciding + formatting
(``select``, ``render``). Folding them together would put a regex translator, a filesystem walk and a
budget allocator in one namespace, and every future change to one would be reviewed against the other
two.

**The one decision this package exists to enforce.** Codna had TWO rules readers that disagreed — a
Python reader that concatenated eight fixed root filenames and truncated at 16 KB, and the vendored
SDK's "root AGENTS.md plus every ``.md`` beside it". A repository's conventions must not depend on
which of Codna's surfaces observed them. So there is now exactly one reader (``discover``), exactly
one applicability decision (``select``), and exactly one formatter (``render``); every consumer —
durable runs, the packaged agent runner, ``codna fix``, ``codna review``, IDE chat, and the
``codna_request_rule`` MCP tool — calls these and nothing else.

**What this package replaces, without regressing it.** ``cli/codna/project_rules.py`` is now a thin
compatibility re-export over :func:`guidance_for_repo`. Its callers (``fix_run``, ``review``, the
packaged runner) keep calling ``read_project_guidance`` and keep getting a string — but that string
now comes from the engine, so a rule that is scoped out is scoped out for them too, and a rule that
does not fit is reported instead of silently sliced away.

Public surface, in the order a consumer needs it:

  * :func:`discover` — find rules (``discovery.discover``)
  * :func:`select`   — decide which apply (``select.select``)
  * :func:`render`   — format within a budget (``render.render``)
  * :func:`load`     — discover + select + render + diagnostics in one call
  * :func:`guidance_for_repo` — the string-only compatibility entry point
  * :func:`request_rule` — pull one rule's body by name (the description-only mechanism)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, TextIO

from .discovery import Discovery, discover, discover_user_rules
from .model import Rule, RuleMode, RuleWarning, Scope
from .render import (
    DEFAULT_MAX_BYTES,
    DroppedRule,
    RenderResult,
    emit_diagnostics,
    render,
    render_guidance,
)
from .select import Selection, select

__all__ = [
    "DEFAULT_MAX_BYTES",
    "Discovery",
    "DroppedRule",
    "RenderResult",
    "Rule",
    "RuleMode",
    "RuleWarning",
    "Scope",
    "Selection",
    "discover",
    "discover_user_rules",
    "emit_diagnostics",
    "guidance_for_repo",
    "load",
    "render",
    "render_guidance",
    "request_rule",
    "select",
]


@dataclass(frozen=True)
class LoadedRules:
    """One turn's rules, with every drop and every read-time warning still attached."""

    discovery: Discovery
    selection: Selection
    result: RenderResult


def load(
    repo_dir: str | None,
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_rules_dir: str | None = None,
    include_user: bool = True,
    include_catalog: bool = True,
) -> LoadedRules:
    """Discover, select and render in one call. Pure apart from reading the repository."""
    discovery = discover(repo_dir, user_rules_dir=user_rules_dir, include_user=include_user)
    selection = select(discovery.rules, files_in_context, explicit_mentions)
    result = render(selection, max_bytes=max_bytes, include_catalog=include_catalog)
    return LoadedRules(discovery=discovery, selection=selection, result=result)


def guidance_for_repo(
    repo_dir: str | None,
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_rules_dir: str | None = None,
    include_user: bool = True,
    stream: TextIO | None = None,
    report: bool = True,
) -> str | None:
    """The repository's rules as prompt text, or ``None`` when there are none.

    ``None`` (not ``""``) is the historical contract of ``read_project_guidance`` and is preserved:
    several callers test ``if guidance:`` and a repository with no rules is the normal case, not an
    error. What is NEW is that a rule which does not fit is printed to ``stream`` (stderr by default)
    as a structured warning instead of being sliced off the end of the string.
    """
    loaded = load(
        repo_dir,
        files_in_context=files_in_context,
        explicit_mentions=explicit_mentions,
        max_bytes=max_bytes,
        user_rules_dir=user_rules_dir,
        include_user=include_user,
    )
    if report:
        emit_diagnostics(
            loaded.result,
            warnings=loaded.discovery.warnings,
            stream=stream if stream is not None else sys.stderr,
        )
    return loaded.result.text or None


def request_rule(
    repo_dir: str | None,
    name: str,
    *,
    user_rules_dir: str | None = None,
    include_user: bool = True,
) -> Rule | None:
    """Pull one rule by name — the mechanism behind the ``codna_request_rule`` MCP tool.

    This is how a ``description``-only (catalogued) rule's body is obtained. Matching is by the same
    case-folded key that precedence uses, so a name the catalog printed is a name this resolves. The
    first match in precedence order wins, identically to rendering, so pulling a shadowed rule cannot
    return the body that rendering would have refused.

    Returns ``None`` when no rule matches. That is a real answer — the caller names it in the error.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return None
    from .discovery import discover as _discover
    from .select import normalise_mentions

    keys = normalise_mentions((wanted,))
    for rule in _discover(repo_dir, user_rules_dir=user_rules_dir, include_user=include_user).rules:
        if rule.name_key() in keys:
            return rule
    return None
