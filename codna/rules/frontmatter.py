"""YAML frontmatter, with the same PyYAML-else-fallback posture as ``codna.yaml``.

``config_file._load`` (``cli/codna/config_file.py:58``) accepts PyYAML when it is installed and falls
back to a strict stdlib parse otherwise, because the CLI must keep working on a machine where the
optional dependency is absent. Rule frontmatter is the same situation — ``.mdc`` files are read on
every turn of every run — so it uses the same two-tier posture, with one addition the config loader
does not need: **a fallback parse must be honest.**

A whole-file JSON fallback works for ``codna.yaml`` because the entire document is the mapping. For
frontmatter the mapping is only the first block and the rule body follows it, so "try JSON" is not
available. The fallback here therefore parses a documented YAML SUBSET (scalars, quoted strings,
block/flow lists, booleans, nulls) and raises :class:`FrontmatterError` on anything outside it.
Refusing is the correct answer: a rule whose ``globs`` were silently mis-parsed into nothing would
look manual and never apply, which is precisely the silent drop this engine exists to remove.

Frontmatter is accepted with either ``---`` or ``+++`` fences (the latter is what some teams arrive
with from other tools). A file without a leading fence has no frontmatter and is returned whole.
"""
from __future__ import annotations

import json
import re

from .globs import split_top_level_commas as _split_top_level_commas

__all__ = ["FrontmatterError", "parse_markdown"]

_FENCE = ("---", "+++")
_KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(.*)$")
# YAML's own decimal forms. Deliberately narrow: `0x1f`, `1e3`, `.inf`, `1.2.3` are left as strings
# because converting some-but-not-all of them is a subtler disagreement than converting none.
_INTEGER = re.compile(r"[-+]?(?:0|[1-9][0-9]*)")
_FLOAT = re.compile(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)")


class FrontmatterError(ValueError):
    """Frontmatter that cannot be read unambiguously. Carries the source path for the caller."""

    def __init__(self, message: str, *, source_path: str) -> None:
        super().__init__(f"{source_path}: {message}")
        self.source_path = source_path


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith("\ufeff") else text


def _split(text: str) -> tuple[str, str] | None:
    """Split a fenced header from the body. Returns ``None`` when there is no header."""
    text = _strip_bom(text)
    lines = text.splitlines(keepends=True)
    if not lines:
        return None
    first = lines[0].strip()
    if first not in _FENCE:
        return None
    fence = first
    for index in range(1, len(lines)):
        if lines[index].strip() == fence:
            header = "".join(lines[1:index])
            return header, "".join(lines[index + 1:])
    return None  # an unterminated fence is not frontmatter; the file is body, and callers report it


def _scalar(raw: str, *, source_path: str = ""):
    """Parse one YAML scalar.

    A double-quoted value is decoded as JSON because that is YAML's own escapes (``\\n``, ``\\"``), but
    a malformed one must surface as :class:`FrontmatterError` — the type every caller already handles.
    Letting ``json.JSONDecodeError`` escape would be the worst outcome: it is not a ``FrontmatterError``,
    so it would travel straight through discovery into a caller that reads rules on every turn.
    """
    value = raw.strip()
    if not value:
        return ""
    quoted = len(value) >= 2 and value[0] in "'\"" and value[-1] in "'\""
    if quoted and value[0] == value[-1]:
        quote, inner = value[0], value[1:-1]
        if quote == "'":
            return inner.replace("''", "'")
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise FrontmatterError(
                f"invalid double-quoted string {value!r}: {exc}", source_path=source_path,
            ) from exc
    if quoted:
        # Opens with one quote and closes with the other: ambiguous, so refuse rather than guess.
        raise FrontmatterError(
            f"mismatched quotes in {value!r}", source_path=source_path,
        )
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "~"):
        return None
    # Numbers, because YAML reads them as numbers and a fallback that returned "42" for `n: 42` would
    # make the two parsers disagree about the same document. Only the unambiguous integer/float forms
    # are converted; anything else stays a string, which is what YAML does with e.g. `1.2.3`.
    if _INTEGER.fullmatch(value):
        try:
            return int(value, 10)
        except ValueError:  # pragma: no cover - a regex-valid integer always parses
            return value
    if _FLOAT.fullmatch(value):
        try:
            return float(value)
        except ValueError:  # pragma: no cover
            return value
    return value


def _strip_comment(value: str) -> str:
    """Drop a trailing `` # comment`` that is not inside quotes."""
    out: list[str] = []
    quote: str | None = None
    for index, ch in enumerate(value):
        if quote is None and ch in "'\"":
            quote = ch
        elif quote is not None and ch == quote:
            quote = None
        elif quote is None and ch == "#" and (index == 0 or value[index - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _fallback_parse(header: str, *, source_path: str) -> dict[str, object]:
    """Parse the documented YAML subset. Raises on anything it cannot read unambiguously."""
    data: dict[str, object] = {}
    pending_key: str | None = None
    pending_list: list[object] | None = None

    def flush() -> None:
        nonlocal pending_key, pending_list
        if pending_key is not None:
            # A bare `key:` with no items is `None` (YAML's own reading), NOT a dropped key: silently
            # omitting it would make the two parsers disagree about the same file, which it is this
            # module's contract to prevent.
            data[pending_key] = pending_list
        pending_key, pending_list = None, None

    for raw_line in header.splitlines():
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if pending_key is None:
                raise FrontmatterError(
                    "list item outside a key (PyYAML is not installed; a list must follow '<key>:' "
                    "with a '-' item on each following indented line)", source_path=source_path,
                )
            if pending_list is None:
                pending_list = []
            pending_list.append(_scalar(stripped[2:]))
            continue
        if line[0] in " \t" and pending_key is not None and not stripped.startswith("- "):
            raise FrontmatterError(
                f"nested mapping under {pending_key!r} needs PyYAML", source_path=source_path,
            )
        if line[0] in " \t":
            raise FrontmatterError("indented line without a key", source_path=source_path)
        flush()
        match = _KEY_LINE.match(stripped)
        if match is None:
            raise FrontmatterError(
                f"cannot parse frontmatter line {stripped!r} (PyYAML is not installed; only "
                "'key: value', quoted scalars, booleans, nulls and '-' lists are supported)",
                source_path=source_path,
            )
        key, raw_value = match.group(1), _strip_comment(match.group(2))
        if raw_value.startswith("["):
            if not raw_value.endswith("]"):
                # An unterminated flow list would become one pattern that matches nothing, turning a
                # scoped rule into a silently-dead one. Refuse instead of guessing.
                raise FrontmatterError(
                    f"unterminated flow list for {key!r}: {raw_value!r}", source_path=source_path,
                )
            inner = raw_value[1:-1].strip()
            # Split at TOP-LEVEL commas only: `{src,tests}/**` is ONE pattern, and a naive split would
            # break it into two that match nothing — a glob-scoped rule silently ceasing to apply.
            data[key] = (
                [] if not inner
                else [_scalar(part, source_path=source_path) for part in _split_top_level_commas(inner)]
            )
            continue
        if not raw_value:
            pending_key, pending_list = key, None
            continue
        data[key] = _scalar(raw_value)
        # A bare key with an empty value must not linger and swallow the next key's list.
    flush()
    return data


def _yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return None
    return yaml


def parse_markdown(text: str, *, source_path: str) -> tuple[dict[str, object], str, bool]:
    """Split ``text`` into ``(metadata, body, had_frontmatter)``.

    ``had_frontmatter`` is the distinction the migration depends on: a file that merely LOOKS like a
    rule and carries no frontmatter keeps its historical always-on behaviour, and the caller decides
    that from this flag rather than by re-reading the file.

    Raises :class:`FrontmatterError` for a header that cannot be parsed, because a rule with a
    half-understood ``globs`` key is a rule that silently stops applying.
    """
    text = _strip_bom(text)
    split = _split(text)
    if split is None:
        return {}, text, False
    header, body = split
    if not header.strip():
        return {}, body.lstrip("\n"), True

    yaml = _yaml()
    if yaml is not None:
        try:
            loaded = yaml.safe_load(header)
        except yaml.YAMLError as exc:
            raise FrontmatterError(f"cannot parse frontmatter: {exc}", source_path=source_path) from exc
        if loaded is None:
            return {}, body.lstrip("\n"), True
        if not isinstance(loaded, dict):
            raise FrontmatterError(
                "frontmatter must be a mapping of keys to values", source_path=source_path,
            )
        metadata = dict(loaded)
    else:
        metadata = _fallback_parse(header, source_path=source_path)
    return metadata, body.lstrip("\n"), True
