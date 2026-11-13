"""The stdlib-only YAML reader for `.codna/policy.yaml`, used when PyYAML is not installed.

WHY THIS EXISTS. `policy_document._yaml_load` prefers PyYAML, and a payload that shipped without it
answered `policy_reader_unavailable` — which the sidecar turns into a refusal of EVERY session
(`403 policy_verdict_unavailable`, including a chat that asks for no policy at all). One absent wheel
took the product down. This module makes that packaging gap degrade to a READER that still works for
the documents a policy actually is, and to a CODED REFUSAL that names the construct it could not read
for anything else. It never guesses and never widens a permission.

TWO READERS, ONE MEANING. Both paths end in the same schema validator, so both consumers of the
verdict still agree. What must not happen is the two PARSERS disagreeing about a document, so the
subset below is deliberately the part of YAML a policy uses, and everything outside it is refused
rather than approximated:

    version: 1                      block mappings, `key: value` and `key:` + a nested block
    tools:                          block sequences (`- item`)
      allow: [read_files, editor]   flow sequences of scalars (`[a, b]`)
      deny: []                      plain, single-quoted and double-quoted scalars
    budget:                         integers, floats, booleans, null
      max_cost_usd: 5.0             full-line and trailing comments
    required_approvals: [run_commands]

REFUSED, in two distinct classes — the distinction is the diagnosis:

* :class:`YamlSyntaxError` — the text is not YAML at all (tab indentation, an unterminated quote or
  flow sequence, content at the wrong indentation). Reported as `policy_unparseable`, exactly as
  PyYAML's own scanner would.
* :class:`UnsupportedYamlError` — the text IS YAML, but outside the subset (anchors, aliases, tags,
  block scalars, flow mappings, nested flow collections, complex keys). Reported as
  `policy_reader_subset_exceeded`, because calling valid YAML "unparseable" would be a lie and
  silently approximating it would be the fail-open this whole file exists to prevent.

DUPLICATE KEYS ARE REFUSED HERE TOO, by :class:`DuplicateKeyError`, because the same fail-open applies
to both readers: PyYAML resolves a repeated `tools:` last-wins, so a second block appended below a
reviewed one would silently replace it. An alias-based document could hide the same replacement, which
is a second reason aliases are refused rather than expanded.
"""
from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "DuplicateKeyError",
    "UnsupportedYamlError",
    "YamlSyntaxError",
    "YamlSubsetError",
    "loads",
]

#: YAML 1.1 booleans, which is what PyYAML's `SafeLoader` resolves. Kept in step so the two readers
#: cannot disagree about `max_cost_usd: yes` (a bool either way, so either way `policy_invalid_type`).
_TRUE = frozenset({"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"})
_FALSE = frozenset({"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"})
_NULL = frozenset({"null", "Null", "NULL", "~"})

#: YAML's own decimal forms, and no others. `0x1f`, `.inf`, `1_000` and `1.2.3` stay strings here,
#: because converting some-but-not-all of them is a subtler disagreement than converting none.
_INTEGER = re.compile(r"[-+]?(?:0|[1-9][0-9]*)\Z")
_FLOAT = re.compile(
    r"[-+]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)\Z"
)
#: A policy key is a plain name. A quoted or complex key is refused: the schema is a fixed key set, so
#: an exotic spelling of `tools` is a document nobody reviewed, not a document to normalize.
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
#: Constructs that are valid YAML indicators and outside the subset. Refused by name so the message can
#: say what was found rather than "invalid".
_INDICATORS = {
    "&": "an anchor",
    "*": "an alias",
    "!": "a tag",
    "|": "a literal block scalar",
    ">": "a folded block scalar",
    "{": "a flow mapping",
    "}": "a flow mapping",
    "%": "a directive",
    "@": "a reserved indicator",
    "`": "a reserved indicator",
    "?": "a complex key",
    ",": "a flow collection",
}


class YamlSubsetError(ValueError):
    """Base class: a document this reader will not read. Always carries a line number."""

    def __init__(self, message: str, lineno: int) -> None:
        super().__init__(f"line {lineno}: {message}")
        self.lineno = lineno
        self.message = message


class YamlSyntaxError(YamlSubsetError):
    """Not YAML. Reported as `policy_unparseable`, the same verdict PyYAML reaches."""


class UnsupportedYamlError(YamlSubsetError):
    """Valid YAML outside the subset. Reported as `policy_reader_subset_exceeded`."""


class DuplicateKeyError(ValueError):
    """A repeated mapping key. Refused rather than last-wins, by BOTH readers."""

    def __init__(self, key: str, lineno: int) -> None:
        super().__init__(f"line {lineno}: duplicate key {key!r}")
        self.key = key
        self.lineno = lineno


class _Line:
    """One significant source line: its indentation, its content, and where it came from."""

    __slots__ = ("indent", "text", "number")

    def __init__(self, indent: int, text: str, number: int) -> None:
        self.indent = indent
        self.text = text
        self.number = number


def loads(text: str) -> Any:
    """Parse the policy YAML subset. Returns the same shapes PyYAML would for those documents."""
    lines = _significant(text)
    if not lines:
        return None
    if lines[0].indent != 0:
        raise YamlSyntaxError("the document must start at column 0", lines[0].number)
    value, index = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise YamlSyntaxError("unexpected content after the document", lines[index].number)
    return value


# ── lexical layer ───────────────────────────────────────────────────────────────────────────────


def _significant(text: str) -> list[_Line]:
    """Drop blank lines and comments, and refuse anything whose INDENTATION is not plain spaces."""
    out: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip(" \t")
        leading = raw[: len(raw) - len(stripped)]
        if "\t" in leading:
            # PyYAML's scanner refuses this too, so it is a SYNTAX error rather than a subset gap.
            raise YamlSyntaxError("a tab used for indentation (illegal in YAML)", number)
        content = _strip_comment(stripped)
        if not content:
            continue
        out.append(_Line(len(leading), content, number))
    return out


def _strip_comment(text: str) -> str:
    """Remove a `#` comment, which YAML starts at column 0 or after whitespace, never inside quotes.

    A BACKSLASH ESCAPES THE NEXT CHARACTER INSIDE A DOUBLE-QUOTED SCALAR, so `"a\\" # c"` is one quoted
    scalar containing `a"` followed by a comment. Tracking only the quote character read the `"` after the
    backslash as the closing delimiter, re-opened on the real closing quote, and then treated ` # c` as
    quoted text — leaving the comment attached, which made `_quoted`'s `json.loads` fail and reported VALID
    YAML as `policy_unparseable`. Single-quoted scalars have no backslash escapes (a quote is doubled
    instead), so the flag is only set for `"`.
    """
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            # The character after a backslash is data, whatever it is.
            escaped = False
            continue
        if quote is not None:
            if quote == '"' and char == "\\":
                escaped = True
                continue
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "#" and (index == 0 or text[index - 1] == " "):
            return text[:index].rstrip()
    return text.rstrip()


# ── block layer ────────────────────────────────────────────────────────────────────────────────


def _parse_block(lines: list[_Line], index: int, indent: int) -> tuple[Any, int]:
    """Parse the block starting at `index`, whose entries are at `indent`. Returns (value, next index)."""
    line = lines[index]
    if line.text == "-" or line.text.startswith("- "):
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[_Line], index: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines) and lines[index].indent == indent:
        line = lines[index]
        if line.text == "-" or line.text.startswith("- "):
            raise YamlSyntaxError("a sequence item where a mapping entry was expected", line.number)
        key, rest = _split_key(line.text, line.number)
        if key in mapping:
            raise DuplicateKeyError(key, line.number)
        value, index = _value_after_key(lines, index + 1, indent, rest, line.number)
        mapping[key] = value
    if index < len(lines) and lines[index].indent > indent:
        raise YamlSyntaxError("unexpected indentation", lines[index].number)
    return mapping, index


def _value_after_key(
    lines: list[_Line], index: int, indent: int, rest: str, lineno: int
) -> tuple[Any, int]:
    """The value for a `key:` entry: inline when `rest` is non-empty, else the nested block if any."""
    if rest:
        return _scalar(rest, lineno), index
    if index < len(lines) and lines[index].indent > indent:
        return _parse_block(lines, index, lines[index].indent)
    return None, index


def _parse_sequence(lines: list[_Line], index: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines) and lines[index].indent == indent:
        line = lines[index]
        if not (line.text == "-" or line.text.startswith("- ")):
            break
        rest = line.text[1:].strip()
        index += 1
        if not rest:
            if index < len(lines) and lines[index].indent > indent:
                value, index = _parse_block(lines, index, lines[index].indent)
            else:
                value = None
            items.append(value)
            continue
        found = _maybe_key(rest, line.number)
        if found is None:
            items.append(_scalar(rest, line.number))
            continue
        key, key_rest = found
        # WHERE THE KEY ACTUALLY STARTS, which is what continuation keys must align with. `rest` is
        # stripped, so the offset has to be measured on the raw text: `- name: a` puts the key at
        # `indent + 2`, `-   name: a` at `indent + 4`. Assuming `indent + 2` read only the one-space
        # spelling, so `-   name: a` / `      other: b` — which PyYAML accepts — was refused as
        # "unexpected indentation" (measured: `yaml.safe_load` yields `{name: a, other: b}`).
        after_dash = line.text[1:]
        key_indent = indent + 1 + (len(after_dash) - len(after_dash.lstrip(" ")))
        item, index = _inline_mapping(lines, index, key_indent, key, key_rest, line.number)
        items.append(item)
    if index < len(lines) and lines[index].indent > indent:
        raise YamlSyntaxError("unexpected indentation", lines[index].number)
    return items, index


def _inline_mapping(
    lines: list[_Line], index: int, indent: int, key: str, rest: str, lineno: int
) -> tuple[dict[str, Any], int]:
    """`- key: value` — the item is a mapping whose further entries align under `key`.

    `indent` is the COLUMN OF THE KEY, not the sequence's dash. The caller measures it from the raw line
    (`_parse_sequence`), because YAML allows any spacing after the dash: `- name: a` and `-   name: a`
    are the same mapping, and a nested value under the first key is indented relative to the KEY.
    """
    first, index = _value_after_key(lines, index, indent, rest, lineno)
    item: dict[str, Any] = {key: first}
    key_indent = indent
    if index < len(lines) and lines[index].indent == key_indent:
        more, index = _parse_mapping(lines, index, key_indent)
        for nested_key, nested_value in more.items():
            if nested_key in item:
                raise DuplicateKeyError(nested_key, lineno)
            item[nested_key] = nested_value
    return item, index


def _split_key(text: str, lineno: int) -> tuple[str, str]:
    found = _maybe_key(text, lineno)
    if found is None:
        raise YamlSyntaxError(
            "a line that is neither a `key: value` entry nor a sequence item", lineno
        )
    return found


def _maybe_key(text: str, lineno: int) -> tuple[str, str] | None:
    """`(key, rest)` for a `key:` / `key: value` line, or None when the line is not a mapping entry."""
    quote: str | None = None
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char != ":":
            continue
        following = text[index + 1 : index + 2]
        if following not in ("", " "):
            continue
        key = text[:index].strip()
        if not _PLAIN_KEY.match(key):
            # A quoted or otherwise exotic key is valid YAML this reader does not implement.
            raise UnsupportedYamlError(f"the mapping key {key!r}, which is not a plain name", lineno)
        return key, text[index + 1 :].strip()
    return None


# ── scalar layer ───────────────────────────────────────────────────────────────────────────────


def _scalar(text: str, lineno: int) -> Any:
    text = text.strip()
    if not text:
        return None
    first = text[0]
    if first == "[":
        return _flow_sequence(text, lineno)
    if first in ("'", '"'):
        return _quoted(text, lineno)
    if first in _INDICATORS:
        raise UnsupportedYamlError(f"{_INDICATORS[first]} ({text[:24]!r})", lineno)
    if first == "]" or first == ":":
        raise YamlSyntaxError(f"a stray {first!r}", lineno)
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    if text in _NULL:
        return None
    if _INTEGER.match(text):
        return int(text)
    if _FLOAT.match(text):
        return float(text)
    return text


def _flow_sequence(text: str, lineno: int) -> list[Any]:
    if not text.endswith("]"):
        raise YamlSyntaxError("an unterminated flow sequence", lineno)
    inner = text[1:-1].strip()
    if not inner:
        return []
    items: list[Any] = []
    for part in _split_top_level(inner, lineno):
        entry = part.strip()
        if not entry:
            raise YamlSyntaxError("an empty flow-sequence entry", lineno)
        if entry[0] in ("[", "{"):
            raise UnsupportedYamlError("a nested flow collection", lineno)
        items.append(_scalar(entry, lineno))
    return items


def _split_top_level(inner: str, lineno: int) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    for index, char in enumerate(inner):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char in ("[", "{"):
            raise UnsupportedYamlError("a nested flow collection", lineno)
        if char == "]":
            raise YamlSyntaxError("a stray ']'", lineno)
        if char == ",":
            parts.append(inner[start:index])
            start = index + 1
    if quote is not None:
        raise YamlSyntaxError("an unterminated quote in a flow sequence", lineno)
    parts.append(inner[start:])
    return parts


def _quoted(text: str, lineno: int) -> Any:
    quote = text[0]
    if len(text) < 2 or text[-1] != quote:
        raise YamlSyntaxError("an unterminated quoted scalar, or content after a quoted scalar", lineno)
    body = text[1:-1]
    if quote == "'":
        # YAML escapes a single quote inside a single-quoted scalar by doubling it.
        return body.replace("''", "'")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # YAML's double-quoted escapes are a superset of JSON's (`\\e`, `\\N`, `\\_`, …). Refusing an
        # escape this reader would have to interpret is the only safe answer.
        raise UnsupportedYamlError(
            "a double-quoted scalar with an escape this reader does not implement", lineno
        ) from None
