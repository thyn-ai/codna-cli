"""Extract a repository into structural code units for Codna's local code memory.

Codna has no symbol extractor of its own — the Algenta engine does that server-side. For the
in-process ``codna memory`` capability, Python uses stdlib ``ast`` and the common code languages use
one generic tree-sitter seam when the packaged grammar wheels are installed.

A :class:`CodeUnit` is a structural unit (not a fixed-size chunk): structure gives cleaner retrieval
and provenance. ``id`` is a stable symbol path so a changed symbol re-indexes with ``upsert`` (no
duplicate row) and a removed symbol can be ``delete``d.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Iterator, Protocol

# The filter columns the memory collection declares. ``CodeUnit.metadata()`` returns EXACTLY these
# (plus the physical partition key ``scope_key``), so Telys' ingest never KeyErrors on a missing column.
FILTER_COLUMNS = ["repo_id", "service", "language", "path", "symbol_type", "commit_sha", "branch"]

# Symbol types emitted day-1 (a subset of INTEGRATION-CODNA §4; error-strings/config/routes come later).
SYMBOL_TYPES = ("module", "class", "function", "method", "test")

_MAX_UNIT_CHARS = 2000  # bound the indexed text slice per symbol

_DEFAULT_IGNORE = (
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".codna-memory",
)


@dataclass(frozen=True)
class CodeUnit:
    """One structural code unit with provenance."""

    id: str            # "<repo_id>:<relpath>:<Qualified.Name>" — stable -> upsert/delete by id
    text: str          # what gets embedded: qualname + docstring + a bounded source slice
    path: str          # repo-relative path
    language: str
    symbol_type: str   # one of SYMBOL_TYPES
    qualname: str
    service: str | None = None

    def metadata(self, repo_id: str, scope: str, *, commit_sha: str = "", branch: str = "") -> dict:
        """Exactly the collection key + ``FILTER_COLUMNS`` (Telys reads each by name)."""
        return {
            "scope_key": scope,
            "repo_id": repo_id,
            "service": self.service or "",
            "language": self.language,
            "path": self.path,
            "symbol_type": self.symbol_type,
            "commit_sha": commit_sha,
            "branch": branch,
        }


class LanguageExtractor(Protocol):
    extensions: tuple[str, ...]
    language: str

    def extract(self, repo_id: str, relpath: str, source: str) -> Iterator[CodeUnit]:
        ...


@dataclass
class ExtractStats:
    files: int = 0       # source files scanned for the requested languages
    units: int = 0       # structural units produced
    skipped: int = 0     # files that failed to parse (e.g. SyntaxError) — counted, not fatal


class UnsupportedLanguageError(ValueError):
    """Raised when a caller explicitly requests a language with no registered extractor."""


def _is_test(relpath: str, name: str) -> bool:
    base = os.path.basename(relpath)
    return base.startswith("test_") or base.endswith("_test.py") or name.startswith("test_")


# The stdlib's ast.get_source_segment re-splits the WHOLE source on every call (it calls
# _splitlines_no_ff(source, maxlines=node.end_lineno) internally). That made segment extraction
# O(symbols x lines) per file — ~98% of Python-extraction wall time on real repos. Instead we split
# once per file and slice the pre-split lines per node. _LINE_SPLIT mirrors ast._line_pattern
# EXACTLY (same alternatives, same DOTALL) and _segment mirrors get_source_segment's slicing rules
# (byte-offset columns, unpadded), so the emitted unit text is byte-identical to before — a
# regression here would silently change embeddings and content hashes for every indexed symbol.
_LINE_SPLIT = re.compile(r"(.*?(?:\r\n|\n|\r|$))", re.DOTALL)


def _segment(lines: list[str], node) -> str:
    """The source slice for ``node`` from pre-split ``lines`` (== ast.get_source_segment, unpadded)."""
    try:
        if node.end_lineno is None or node.end_col_offset is None:
            return ""
        lineno = node.lineno - 1
        end_lineno = node.end_lineno - 1
        col_offset = node.col_offset
        end_col_offset = node.end_col_offset
    except AttributeError:
        return ""
    if end_lineno == lineno:
        return lines[lineno].encode()[col_offset:end_col_offset].decode()
    first = lines[lineno].encode()[col_offset:].decode()
    last = lines[end_lineno].encode()[:end_col_offset].decode()
    return first + "".join(lines[lineno + 1:end_lineno]) + last


class PythonExtractor:
    """Python structural units via stdlib ``ast`` — no third-party parser."""

    extensions = (".py",)
    language = "python"

    def extract(self, repo_id: str, relpath: str, source: str) -> Iterator[CodeUnit]:
        tree = ast.parse(source)   # may raise SyntaxError; extract_repo counts the skip (non-fatal)
        doc = ast.get_docstring(tree)
        lines = _LINE_SPLIT.findall(source)   # split ONCE; every symbol slice reuses this
        if doc:
            yield self._unit(repo_id, relpath, relpath, f"{relpath}\n{doc}", "module")
        yield from self._walk(repo_id, relpath, lines, tree.body, prefix="")

    def _walk(self, repo_id, relpath, lines, body, prefix) -> Iterator[CodeUnit]:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{node.name}"
                if _is_test(relpath, node.name):
                    sym = "test"
                elif prefix:
                    sym = "method"
                else:
                    sym = "function"
                yield self._unit(repo_id, relpath, qual, self._body_text(lines, node, qual), sym)
            elif isinstance(node, ast.ClassDef):
                qual = f"{prefix}{node.name}"
                yield self._unit(repo_id, relpath, qual, self._header_text(lines, node, qual), "class")
                yield from self._walk(repo_id, relpath, lines, node.body, prefix=f"{qual}.")

    def _body_text(self, lines, node, qual) -> str:
        seg = _segment(lines, node)
        doc = ast.get_docstring(node) or ""
        return f"{qual}\n{doc}\n{seg}".strip()[:_MAX_UNIT_CHARS]

    def _header_text(self, lines, node, qual) -> str:
        # for a class, prefer the signature line + docstring over the whole body
        seg = _segment(lines, node)
        first = seg.splitlines()[0] if seg else ""
        doc = ast.get_docstring(node) or ""
        return f"{qual}\n{first}\n{doc}".strip()[:_MAX_UNIT_CHARS]

    def _unit(self, repo_id, relpath, qual, text, symbol_type) -> CodeUnit:
        return CodeUnit(
            id=f"{repo_id}:{relpath}:{qual}",
            text=text or qual,
            path=relpath,
            language=self.language,
            symbol_type=symbol_type,
            qualname=qual,
        )


_EXTRACTORS: dict[str, LanguageExtractor] = {}
_EXTRACTOR_ERRORS: dict[str, str] = {}


def register_extractor(extractor: LanguageExtractor) -> None:
    """Register a language extractor by its file extensions (the seam for Go/TS/… later)."""
    for ext in extractor.extensions:
        _EXTRACTORS[ext] = extractor


def registered_languages() -> tuple[str, ...]:
    """Return every language currently backed by a registered extractor."""
    return tuple(sorted({extractor.language for extractor in _EXTRACTORS.values()}))


def language_for(relpath: str) -> str | None:
    """Language a path would be extracted as, or None when no extractor covers its extension.

    For callers that must classify a handful of files (e.g. a CI diff's changed paths) without
    paying for a full ``extract_repo`` walk."""
    extractor = _EXTRACTORS.get(os.path.splitext(relpath)[1])
    return extractor.language if extractor is not None else None


def extract_file(repo_id: str, relpath: str, source: str) -> list[CodeUnit]:
    """Extract ONE file with the extractor registered for its extension.

    Returns [] when no extractor covers the extension. Parse errors propagate — unlike
    ``extract_repo``, which skips and counts them."""
    extractor = _EXTRACTORS.get(os.path.splitext(relpath)[1])
    if extractor is None:
        return []
    return list(extractor.extract(repo_id, relpath, source))


def _record_extractor_error(language: str, message: str) -> None:
    _EXTRACTOR_ERRORS[language] = message


def resolve_languages(languages: tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve and validate the requested extractor languages."""
    available = set(registered_languages())
    resolved = tuple(registered_languages() if languages is None else languages)
    missing = sorted(set(resolved) - available)
    if missing:
        available_text = ", ".join(sorted(available)) if available else "none"
        missing_text = ", ".join(missing)
        diagnostics = [f"{lang}: {_EXTRACTOR_ERRORS[lang]}" for lang in missing if lang in _EXTRACTOR_ERRORS]
        if "tree-sitter" in _EXTRACTOR_ERRORS:
            diagnostics.append(f"tree-sitter: {_EXTRACTOR_ERRORS['tree-sitter']}")
        details = f" Registration failures: {'; '.join(diagnostics)}." if diagnostics else ""
        raise UnsupportedLanguageError(
            f"unsupported language extractor(s): {missing_text}. "
            f"Registered languages: {available_text}.{details} "
            "Reinstall `codna`, or use the backward-compatible `codna[memory-languages]` extra, "
            "for JS/TS/Go/Rust/Java/C/C++/C#/PHP/Ruby."
        )
    return resolved


register_extractor(PythonExtractor())

# Multi-language support (additive): plug tree-sitter grammars into the same seam when the wheels are
# installed. Base wheels declare them; source/dev installs still report explicit diagnostics if absent.
try:
    from .codeunits_treesitter import register_treesitter

    register_treesitter(register_extractor, CodeUnit, record_error=_record_extractor_error)
except Exception as exc:  # noqa: BLE001 — tree-sitter is optional, but the failure must remain inspectable.
    _record_extractor_error("tree-sitter", f"{type(exc).__name__}: {exc}")


def extract_repo(repo_path: str, repo_id: str, *, languages: tuple[str, ...] | None = None,
                 ignore: tuple[str, ...] = _DEFAULT_IGNORE) -> tuple[list[CodeUnit], ExtractStats]:
    """Walk ``repo_path`` and return (units, stats). Files that fail to parse are skipped + counted."""
    langs = set(resolve_languages(languages))
    units: list[CodeUnit] = []
    stats = ExtractStats()
    ignore_set = set(ignore)
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_set and not d.startswith(".")]
        for fn in files:
            ext = os.path.splitext(fn)[1]
            ex = _EXTRACTORS.get(ext)
            if ex is None or ex.language not in langs:
                continue
            full = os.path.join(root, fn)
            relpath = os.path.relpath(full, repo_path)
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except OSError:
                continue
            if "\x00" in source:
                # Some source files carry embedded NUL bytes (binary-ish blobs, generated code, certain repos
                # — caveman/tailwindcss/grpc/Ventoy/TrafficMonitor). The on-device embedder rejects NUL, so
                # strip it here at extraction (a NUL is never semantically meaningful in source text).
                source = source.replace("\x00", "")
            stats.files += 1
            try:
                got = list(ex.extract(repo_id, relpath, source))
            except SyntaxError:
                stats.skipped += 1
                continue
            units.extend(got)
            stats.units += len(got)
    return units, stats
