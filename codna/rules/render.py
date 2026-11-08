"""Deterministic rendering: precedence, budget, and a report of what did not fit.

Rendering is where the old defect lived, so this module is written to make that defect structurally
impossible. The old reader concatenated every file then sliced the string with ``[:MAX_BYTES]``,
which silently cut the tail of whichever file sorted last — a user could add a rule and watch it
disappear with no error and no way to notice.

Here a rule is either rendered IN FULL, or it is REPORTED as dropped (name, path, byte cost), or —
only when it alone exceeds the entire budget — it is truncated and the truncation is REPORTED. There
is no fourth outcome, and nothing is ever cut without a warning naming it.

**Order.** ``Rule.sort_key()`` — precedence scope (Team → Project → User), then specificity
(most-nested first), then path, then name. The ordering is total, so two runs over one tree produce
byte-identical output.

**Precedence.** Two rules conflict when their names match case-insensitively. The FIRST in render
order wins, and because render order begins with the scope rank, the winner is decided by SOURCE —
a Team rule beats a Project rule with the same name regardless of filename or directory sort order.
The loser is returned as ``DroppedRule(reason="precedence")`` so a shadowed rule is never invisible.

**Budget.** Rules are taken in render order while the accumulated bytes fit. A rule that does not fit
is skipped and the scan CONTINUES, so one oversized rule cannot starve a smaller one after it. Because
the order is deterministic, so is the kept set. Every skip is recorded with its exact cost.

**Catalog lines.** Description-only rules are appended as one line each, carrying name, description
and globs — enough to decide to pull the body via ``codna_request_rule`` — and they count against the
same budget, because they are the same prompt.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, TextIO

from .model import Rule, RuleWarning
from .select import Selection

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DroppedRule",
    "RenderResult",
    "TruncatedRule",
    "emit_diagnostics",
    "render",
    "render_guidance",
]

#: The historical ceiling, kept as the default so a caller that says nothing gets today's behaviour.
DEFAULT_MAX_BYTES = 16 * 1024

#: Why a rule that applied by mode is not fully present in the text.
DROP_BUDGET = "budget"
DROP_PRECEDENCE = "precedence"

_CATALOG_HEADER = "# RULES AVAILABLE ON REQUEST (pull one by name with codna_request_rule)"

#: Appended when a body had to be cut, so a model never reads a half sentence as a whole rule.
_TRUNCATION_MARKER = "\n\n[rules: this rule was truncated to fit the budget]"


@dataclass(frozen=True)
class DroppedRule:
    """A rule that applied by mode but is absent from the prompt, and why. Never silent."""

    name: str
    source_path: str
    reason: str
    bytes_needed: int

    def as_warning(self) -> RuleWarning:
        if self.reason == DROP_PRECEDENCE:
            detail = "shadowed by a higher-precedence rule with the same name"
        else:
            detail = f"needs {self.bytes_needed} bytes and the budget is exhausted"
        return RuleWarning(
            code=f"rule_dropped_{self.reason}",
            message=detail,
            source_path=self.source_path,
        )


@dataclass(frozen=True)
class TruncatedRule:
    """A rule whose BODY was cut to fit — the last resort, and reported.

    Used only when a single rule is larger than the entire budget AND nothing has been rendered yet.
    The alternative would be returning no guidance at all for a repository that does have rules,
    which is the regression the existing durable test exists to catch. Unlike the old
    ``[:MAX_BYTES]`` slice, this is explicit, bounded to one rule, and announced.
    """

    name: str
    source_path: str
    bytes_kept: int
    bytes_total: int

    def as_warning(self) -> RuleWarning:
        return RuleWarning(
            code="rule_truncated_for_budget",
            message=(
                f"body cut from {self.bytes_total} to {self.bytes_kept} bytes: this rule alone "
                "exceeds the whole rules budget (raise it or split the rule)"
            ),
            source_path=self.source_path,
        )


@dataclass(frozen=True)
class RenderResult:
    """The prompt text plus the complete account of what was and was not included."""

    text: str = ""
    kept: tuple[Rule, ...] = ()
    dropped: tuple[DroppedRule, ...] = ()
    catalogued: tuple[Rule, ...] = ()
    truncated: tuple[TruncatedRule, ...] = ()
    budget_bytes: int = DEFAULT_MAX_BYTES

    @property
    def used_bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def dropped_for_budget(self) -> tuple[DroppedRule, ...]:
        return tuple(d for d in self.dropped if d.reason == DROP_BUDGET)

    @property
    def dropped_for_precedence(self) -> tuple[DroppedRule, ...]:
        return tuple(d for d in self.dropped if d.reason == DROP_PRECEDENCE)

    @property
    def partially_truncated(self) -> bool:
        """True when content was withheld. The caller MUST surface this; it is the old defect."""
        return bool(self.dropped_for_budget or self.truncated)

    def warnings(self) -> tuple[RuleWarning, ...]:
        return tuple(d.as_warning() for d in self.dropped) + tuple(
            t.as_warning() for t in self.truncated
        )


def _block(rule: Rule) -> str:
    """One rule as it appears in the prompt. Byte-counted by its encoded size, not by len()."""
    return f"# {rule.source_path}\n{rule.body}"


def _catalog_line(rule: Rule) -> str:
    scope = f" [{rule.base_dir}]" if rule.base_dir else ""
    detail = rule.description or "no description"
    patterns = ", ".join(rule.globs)
    globs = f" — globs: {patterns}" if patterns else ""
    return f"- {rule.name}{scope}: {detail}{globs}"


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _fit_single(rule: Rule, max_bytes: int) -> tuple[str, TruncatedRule | None]:
    """Render one oversized rule by cutting its BODY on a character boundary that fits."""
    header = f"# {rule.source_path}\n"
    total = _size(_block(rule))
    budget = max_bytes - _size(header) - _size(_TRUNCATION_MARKER)
    if budget <= 0:
        return "", TruncatedRule(rule.name, rule.source_path, 0, total)
    encoded = rule.body.encode("utf-8")
    cut = encoded[:budget]
    # Never emit a half-decoded character: back off to the last complete codepoint.
    while cut:
        try:
            body = cut.decode("utf-8")
            break
        except UnicodeDecodeError:
            cut = cut[:-1]
    else:
        body = ""
    text = header + body + _TRUNCATION_MARKER
    return text, TruncatedRule(rule.name, rule.source_path, _size(text), total)


def render(
    selection: Selection,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    include_catalog: bool = True,
) -> RenderResult:
    """Render ``selection`` within ``max_bytes``, reporting everything excluded.

    ``max_bytes`` is the ONE budget. The durable path's smaller ceiling (8 KB) is passed here as a
    parameter instead of re-truncating the result afterwards, so there is exactly one place a rule can
    be dropped and exactly one place that says which rules were.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be >= 0")

    # Precedence: first in render order wins a name conflict. `selection.selected` is already sorted
    # by `sort_key`, so this loop sees Team before Project before User.
    winners: list[Rule] = []
    dropped: list[DroppedRule] = []
    claimed: dict[str, Rule] = {}
    for rule in selection.selected:
        key = rule.conflict_key()
        if key in claimed:
            dropped.append(DroppedRule(
                name=rule.name,
                source_path=rule.source_path,
                reason=DROP_PRECEDENCE,
                bytes_needed=_size(_block(rule)),
            ))
            continue
        claimed[key] = rule
        winners.append(rule)

    kept: list[Rule] = []
    blocks: list[str] = []
    truncated: list[TruncatedRule] = []
    used = 0
    # Truncation is reserved for the case where it is the ONLY way to answer at all: a single rule
    # larger than the whole budget. When several rules apply, dropping the oversized one is strictly
    # better — a truncated rule would consume the entire budget and starve every smaller rule behind
    # it, which is the same starvation the old blind slice caused.
    only_rule = len(winners) == 1
    for rule in winners:
        block = _block(rule)
        cost = _size(block)
        # +2 for the blank line joining this block to what is already there.
        needed = cost + (2 if blocks else 0)
        if used + needed <= max_bytes:
            blocks.append(block)
            kept.append(rule)
            used += needed
            continue
        if only_rule and not blocks:
            text, record = _fit_single(rule, max_bytes)
            if text and record is not None:
                blocks.append(text)
                kept.append(rule)
                truncated.append(record)
                used += _size(text)
                continue
        dropped.append(DroppedRule(
            name=rule.name,
            source_path=rule.source_path,
            reason=DROP_BUDGET,
            bytes_needed=cost,
        ))

    body = "\n\n".join(blocks)

    catalogued: list[Rule] = []
    if include_catalog and selection.catalogued:
        lines: list[str] = []
        for rule in selection.catalogued:
            line = _catalog_line(rule)
            cost = _size(line) + 1
            separator = 2 if (body or lines) else 0
            if used + cost + separator > max_bytes:
                dropped.append(DroppedRule(
                    name=rule.name,
                    source_path=rule.source_path,
                    reason=DROP_BUDGET,
                    bytes_needed=cost,
                ))
                continue
            lines.append(line)
            catalogued.append(rule)
            used += cost + separator
        if lines:
            catalog = _CATALOG_HEADER + "\n" + "\n".join(lines)
            body = f"{body}\n\n{catalog}" if body else catalog

    return RenderResult(
        text=body.strip(),
        kept=tuple(kept),
        dropped=tuple(dropped),
        catalogued=tuple(catalogued),
        truncated=tuple(truncated),
        budget_bytes=max_bytes,
    )


def render_guidance(
    rules: Iterable[Rule],
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    include_catalog: bool = True,
) -> RenderResult:
    """``select`` then ``render`` — the whole pipeline over already-discovered rules."""
    from .select import select

    return render(
        select(rules, files_in_context, explicit_mentions),
        max_bytes=max_bytes,
        include_catalog=include_catalog,
    )


def emit_diagnostics(
    result: RenderResult,
    *,
    warnings: Iterable[RuleWarning] = (),
    stream: TextIO | None = None,
    label: str = "codna",
) -> None:
    """Write every drop, truncation and read-time warning to ``stream`` (stderr by default).

    This is the half of the fix a caller cannot forget: the result carries the account, and this
    writes it somewhere a user sees. A drop that lives only in a return value is barely less silent
    than the slice it replaced.
    """
    out = stream if stream is not None else sys.stderr
    for warning in warnings:
        print(f"{label}: warning: [rules] {warning.code}: {warning.source_path}: {warning.message}",
              file=out)
    branch_drops = result.dropped_for_budget
    if branch_drops:
        names = ", ".join(f"{d.name} ({d.source_path})" for d in branch_drops)
        print(
            f"{label}: warning: [rules] {len(branch_drops)} rule(s) exceeded the "
            f"{result.budget_bytes}-byte budget and were NOT attached: {names}",
            file=out,
        )
    precedence_drops = result.dropped_for_precedence
    if precedence_drops:
        names = ", ".join(f"{d.name} ({d.source_path})" for d in precedence_drops)
        print(
            f"{label}: warning: [rules] {len(precedence_drops)} rule(s) shadowed by a "
            f"higher-precedence rule with the same name: {names}",
            file=out,
        )
    for record in result.truncated:
        print(f"{label}: warning: [rules] {record.as_warning().code}: {record.source_path}: "
              f"{record.as_warning().message}", file=out)
