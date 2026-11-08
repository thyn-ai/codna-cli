"""Applicability: which rules attach to THIS turn. Pure, total, and side-effect free.

``select`` answers exactly one question — "does this rule apply here?" — and answers it the same way
for every consumer. It has no I/O, reads no clock, and sorts on ``Rule.sort_key()``, so the same
``(rules, files, mentions)`` always produces the same ordered tuple. That is what lets IDE chat and
``codna fix`` agree: they call this one function.

The four modes, and what each means for inclusion:

===============  ==========================================  =============================
mode             included when                               rendered as
===============  ==========================================  =============================
``always``       always (subject to its subtree)             full body
``glob``         a file in context matches one of its globs  full body
``catalog``      the caller explicitly requests it            one catalog line, else
``manual``       the caller explicitly mentions it           full body, else absent
===============  ==========================================  =============================

**Catalogued rules are cost-bounded the same way Cursor bounds them**, and the pull is explicit for
the same reason: a ``description``-only rule exists so the model can decide it needs the rule. Codna
implements that as the ``codna_request_rule`` MCP tool (see ``cli/codna/mcp_server.py``), which
returns one rule's body by name. A tool, not prompt-side guessing: the tool is what makes the request
observable and stateful — the turn asks for a rule, gets its text, and the request is logged — where
"hope the model re-reads the catalog" is neither. A catalog line therefore carries enough to make the
decision (name, description, globs) and never the body.

**Subtree nesting.** A rule discovered under ``pkg/api`` applies only to files under ``pkg/api``. The
one deliberate exception is an EMPTY file set, which means "the caller did not say". Glob rules then
match nothing (correct, and reported); an always-apply rule is kept, because dropping a rule that is
already always-on whenever an optional argument is omitted is a silent regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .model import Rule, RuleMode

__all__ = ["Selection", "normalise_mentions", "normalise_paths", "select"]


def normalise_mentions(mentions: Iterable[str] | None) -> frozenset[str]:
    """Rule names an ``@``-mention refers to, case-folded and without the ``@``.

    Accepts ``"@python"``, ``"python"``, and a bare filename like ``"python.mdc"`` — the last because
    that is what a user sees in the rules directory and typing the extension is a natural mistake
    that must not silently select nothing. Only the RULE extensions are stripped, so a rule legitimately
    named ``db.v2`` keeps its dotted name.
    """
    out: set[str] = set()
    for raw in mentions or ():
        text = str(raw or "").strip()
        if text.startswith("@"):
            text = text[1:]
        text = text.strip()
        if not text:
            continue
        out.add(text.casefold())
        base = text.rsplit("/", 1)[-1]
        for suffix in (".mdc", ".md", ".markdown"):
            if base.casefold().endswith(suffix):
                out.add(base[: -len(suffix)].casefold())
                break
    return frozenset(out)


def normalise_paths(files: Iterable[str] | None) -> tuple[str, ...]:
    """Repository-relative POSIX paths, de-duplicated and order-preserved."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in files or ():
        rel = Rule.normalise_rel(raw)
        if not rel or rel in seen:
            continue
        seen.add(rel)
        ordered.append(rel)
    return tuple(ordered)


@dataclass(frozen=True)
class Selection:
    """The outcome of ``select``, split so nothing that was excluded disappears silently."""

    #: Rules whose bodies render, in deterministic render order.
    selected: tuple[Rule, ...] = ()
    #: Description-only rules the turn did not request: a line each, so the model can ask for one.
    catalogued: tuple[Rule, ...] = ()
    #: Rules that applied by mode but were excluded by their subtree (a nested rule outside it).
    out_of_scope: tuple[Rule, ...] = ()
    #: Glob rules that applied by mode but matched none of the files in context.
    unmatched: tuple[Rule, ...] = ()
    #: The file set the decision was made against (empty when the caller named none).
    files: tuple[str, ...] = ()
    #: Whether the caller actually NAMED a file set. ``False`` means "unknown", not "none", and the
    #: difference decides glob handling — see :func:`select`.
    files_known: bool = False
    #: The mentions the decision was made against, normalised.
    mentions: frozenset[str] = field(default_factory=frozenset)

    @property
    def all_applicable(self) -> tuple[Rule, ...]:
        """Everything that survived mode/subtree filtering, selected and catalogued alike."""
        return self.selected + self.catalogued


def select(
    rules: Sequence[Rule],
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
) -> Selection:
    """Which rules apply for ``files_in_context`` / ``explicit_mentions``. Pure and deterministic.

    **``None`` and empty are different answers, and the difference is load-bearing.**

    * ``None`` — the caller did not say which files this turn touches. A durable run is the real case:
      its agent may edit anything in the tree, so a globbed rule cannot be RULED OUT. Glob rules are
      therefore included, and any that matched nothing are reported as ``unmatched`` so the decision is
      visible. Dropping them here would be silent under-application — the agent would not be told a
      convention that governs the very files it is about to change.
    * ``[]`` — the caller said the set is empty (nothing changed, an empty diff). Nothing can match, so
      glob rules are excluded, which is the correct answer for a review of an empty change.
    """
    files_known = files_in_context is not None
    files = normalise_paths(files_in_context)
    mentions = normalise_mentions(explicit_mentions)

    selected: list[Rule] = []
    catalogued: list[Rule] = []
    out_of_scope: list[Rule] = []
    unmatched: list[Rule] = []

    for rule in sorted(rules, key=lambda r: r.sort_key()):
        requested = rule.name_key() in mentions
        mode = rule.mode

        # A rule's subtree is only knowable when the caller named files. With an UNKNOWN file set a
        # nested rule is kept (dropping it would be a silent loss); with a KNOWN one it must be in play.
        if files_known and not rule.in_subtree(files):
            out_of_scope.append(rule)
            continue

        if mode is RuleMode.ALWAYS:
            selected.append(rule)
        elif mode is RuleMode.GLOB:
            if requested:
                selected.append(rule)
            elif not files_known:
                # Adaptive: an unknown file set cannot rule this out, so include it and report it.
                selected.append(rule)
                unmatched.append(rule)
            elif rule.matches_any_file(files):
                selected.append(rule)
            else:
                unmatched.append(rule)
        elif mode is RuleMode.CATALOG:
            if requested:
                selected.append(rule)
            else:
                catalogued.append(rule)
        else:  # MANUAL
            if requested:
                selected.append(rule)

    return Selection(
        selected=tuple(selected),
        catalogued=tuple(catalogued),
        out_of_scope=tuple(out_of_scope),
        unmatched=tuple(unmatched),
        files=files,
        files_known=files_known,
        mentions=mentions,
    )
