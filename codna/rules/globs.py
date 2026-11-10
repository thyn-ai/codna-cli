"""Glob → matcher, with the ``**`` semantics Cursor's ``globs:`` frontmatter requires.

Three things this module owns, all of them places a naive ``fnmatch`` gets it wrong:

1. **``**`` is a path-segment wildcard.** ``src/**/*.ts`` must match ``src/a.ts`` *and*
   ``src/deep/a.ts``; ``fnmatch`` treats ``**`` as a single-segment ``*`` and misses both the
   zero-directory case and, on some platforms, the separator entirely. ``**`` is translated as
   "zero or more whole segments", never as "any characters".
2. **``*`` never crosses a ``/``.** ``src/*.ts`` matches ``src/a.ts`` and NOT ``src/deep/a.ts``.
3. **Comma-separated lists are supported**, because ``globs: src/**/*.ts, tests/**`` is a shape
   people write and silently losing the second pattern is the failure mode this file exists to
   avoid. Commas are only split at brace depth 0, so ``{a,b}`` stays one pattern.

Brace alternation (``{a,b}``) is expanded before translation rather than compiled into the regex, so
every produced pattern is a plain path glob and the regex stays linear (no nested quantifier that
could backtrack catastrophically on an adversarial pattern).

Matching is CASE-SENSITIVE, like ``picomatch`` and like the repository paths it is applied to; the
only normalisation performed is backslash → slash and the removal of a leading ``./``.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

__all__ = [
    "compile_glob",
    "matches",
    "matches_any",
    "parse_globs",
    "split_top_level_commas",
]


def split_top_level_commas(value: str) -> list[str]:
    """Split on commas that are not inside ``{}``/``[]``.

    ``"{**/*.ts,**/*.tsx}, docs/**"`` splits into ``["{**/*.ts,**/*.tsx}", "docs/**"]`` — the brace
    group is one pattern because it is a single alternative set, not two.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in value:
        if ch in "{[":
            depth += 1
        elif ch in "}]" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def parse_globs(value: object) -> tuple[str, ...]:
    """Normalise a frontmatter ``globs`` value into an ordered, de-duplicated tuple of patterns.

    Accepts a comma-separated string (the documented form), a YAML list, or a single pattern.
    Anything else is rejected rather than coerced: a ``globs: 42`` that silently became no globs
    would turn a scoped rule into a manual one and look like it "just didn't apply".
    """
    patterns: list[str] = []
    if isinstance(value, str):
        patterns.extend(split_top_level_commas(value))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"globs entries must be strings, got {type(item).__name__}")
            patterns.extend(split_top_level_commas(item))
    elif value is None:
        return ()
    else:
        raise ValueError(f"globs must be a string or a list of strings, got {type(value).__name__}")
    seen: set[str] = set()
    out: list[str] = []
    for raw in patterns:
        pattern = raw.strip()
        if not pattern or pattern in seen:
            continue
        seen.add(pattern)
        out.append(pattern)
    return tuple(out)


def _expand_braces(pattern: str) -> list[str]:
    """Expand ``{a,b}`` alternation into concrete patterns (depth-first, order preserved)."""
    start = pattern.find("{")
    if start < 0:
        return [pattern]
    depth = 0
    end = -1
    for index in range(start, len(pattern)):
        if pattern[index] == "{":
            depth += 1
        elif pattern[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:  # unbalanced: treat the '{' literally rather than guessing
        return [pattern]
    prefix, group, suffix = pattern[:start], pattern[start + 1:end], pattern[end + 1:]
    expanded: list[str] = []
    for option in split_top_level_commas(group):
        expanded.extend(_expand_braces(prefix + option + suffix))
    return expanded


def _translate(pattern: str) -> str:
    """Translate one brace-free glob into an anchored regex body."""
    out: list[str] = []
    index = 0
    length = len(pattern)
    while index < length:
        ch = pattern[index]
        if ch == "*":
            run = index
            while run < length and pattern[run] == "*":
                run += 1
            stars = run - index
            at_segment_start = index == 0 or pattern[index - 1] == "/"
            at_segment_end = run == length or pattern[run] == "/"
            if stars >= 2 and at_segment_start and at_segment_end:
                # A `**` segment: zero or more whole path segments. Consume the following slash so
                # `a/**/b` becomes `a/(?:[^/]+/)*b` and therefore also matches `a/b`.
                if run < length:
                    out.append("(?:[^/]+/)*")
                    index = run + 1
                else:
                    out.append(".*")
                    index = run
            else:
                out.append("[^/]*")
                index = run
            continue
        if ch == "?":
            out.append("[^/]")
            index += 1
            continue
        if ch == "[":
            close = pattern.find("]", index + 1)
            if close > index + 1:
                body = pattern[index + 1:close]
                if body.startswith("!"):
                    body = "^" + body[1:]
                # Escape the one character that would terminate the class early.
                out.append("[" + body.replace("\\", "\\\\") + "]")
                index = close + 1
                continue
            out.append(re.escape(ch))
            index += 1
            continue
        out.append(re.escape(ch))
        index += 1
    return "".join(out)


_CACHE: dict[str, re.Pattern[str]] = {}


def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile one glob (brace alternation allowed) into a matcher over a repository-relative path."""
    cached = _CACHE.get(pattern)
    if cached is not None:
        return cached
    normalised = pattern.strip().replace("\\", "/")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    alternatives = [_translate(part) for part in _expand_braces(normalised)]
    compiled = re.compile(r"\A(?:" + "|".join(alternatives) + r")\Z")
    _CACHE[pattern] = compiled
    return compiled


def matches(pattern: str, rel_path: str) -> bool:
    """Whether one glob matches one repository-relative path."""
    path = str(rel_path or "").replace("\\", "/").lstrip("/")
    while path.startswith("./"):
        path = path[2:]
    return bool(compile_glob(pattern).match(path))


def matches_any(patterns: Sequence[str] | Iterable[str], rel_path: str) -> bool:
    """Whether ANY glob in the sequence matches. An empty sequence matches nothing."""
    return any(matches(pattern, rel_path) for pattern in patterns)
