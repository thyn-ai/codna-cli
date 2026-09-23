"""Multi-language structural extraction via tree-sitter — ONE generic walk for every grammar.

Python stays on the stdlib-``ast`` path (:mod:`codna.codeunits`). This module plugs common code
languages into the same ``register_extractor`` seam. Base release wheels install these dependencies;
source/dev installs can still run Python-only if the grammar wheels are absent, and explicit language
requests report diagnostics.

There is NO per-language logic: a node is a unit when its grammar type *contains* a function/type hint
and has a name; the doc is the leading comment sibling(s), else a leading string literal in the body
(docstring languages). Validated across 12 languages on 200+ public repos (recall@10 0.60, and the
WordLlama precision rerank lifts it to 0.66 — broadly, not just Python)."""
from __future__ import annotations

import importlib
import importlib.util
import re
from functools import lru_cache
from typing import Callable

_MAX = 2000

# grammar -> (pip module, capsule accessor). Adding a language = one entry + ``pip install tree-sitter-<x>``.
# Python is intentionally absent: the stdlib-``ast`` PythonExtractor stays the (tested) path for ``.py``.
GRAMMAR_MOD = {
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"), "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"), "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"), "csharp": ("tree_sitter_c_sharp", "language"),
    "php": ("tree_sitter_php", "language_php"), "ruby": ("tree_sitter_ruby", "language"),
}
EXT_LANG = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp", ".php": "php", ".rb": "ruby",
}
_FUNC = ("function", "method", "constructor", "subroutine", "fn_", "_fn", "def")
_TYPE = ("class", "struct", "interface", "trait", "impl", "enum", "module", "namespace", "object_definition")
_NOISE = re.compile(r"SPDX-License|Copyright\s*[\(©]|Licensed under|All rights reserved|"
                    r"GNU General Public|Permission is hereby granted|Redistribution and use", re.I)


@lru_cache(maxsize=64)
def _parser(lang):
    from tree_sitter import Language, Parser
    mod, acc = GRAMMAR_MOD[lang]
    return Parser(Language(getattr(importlib.import_module(mod), acc)()))


def _txt(node, src): return src[node.start_byte:node.end_byte].decode("utf-8", "replace")
def _is_func(t): return any(h in t for h in _FUNC)
def _is_type(t): return any(h in t for h in _TYPE)


def _name_of(node, src):
    n = node.child_by_field_name("name")
    if n is not None:
        return _txt(n, src)
    for c in node.children:
        if "identifier" in c.type or c.type.endswith("name"):
            return _txt(c, src)
    return ""


def _leading_doc(node, src):
    parts, sib = [], node.prev_sibling
    while sib is not None and "comment" in sib.type:
        parts.append(_txt(sib, src))
        sib = sib.prev_sibling
    if parts:
        return "\n".join(reversed(parts))
    body = node.child_by_field_name("body")
    if body is not None:
        for c in list(body.children)[:3]:
            if "string" in c.type:
                return _txt(c, src)
            for g in list(c.children)[:2]:
                if "string" in g.type:
                    return _txt(g, src)
    return ""


def _iter(node, src, prefix, depth, out):
    if depth > 6:
        return
    for c in node.children:
        t = c.type
        if _is_func(t) or _is_type(t):
            name = _name_of(c, src)
            if name and re.match(r"^[A-Za-z_][\w]*$", name):
                qual = f"{prefix}{name}"
                doc = _leading_doc(c, src)
                if _NOISE.search(doc):
                    doc = ""
                is_type = _is_type(t) and not _is_func(t)
                out.append((qual, "class" if is_type else ("method" if prefix else "function"),
                            doc, _txt(c, src)[:_MAX]))
                if is_type:  # descend type bodies for methods, not fn bodies
                    _iter(c.child_by_field_name("body") or c, src, f"{qual}.", depth + 1, out)
        else:
            _iter(c, src, prefix, depth + 1, out)


def _iter_symbols(lang, source):
    src = source.encode("utf-8", "replace")
    out = []
    _iter(_parser(lang).parse(src).root_node, src, "", 0, out)
    return out


def _make_extractor(CodeUnit, lang, exts):
    class _TS:
        extensions = tuple(exts)
        language = lang
        def extract(self, repo_id, relpath, source):
            for qual, sym, doc, body in _iter_symbols(lang, source):
                text = f"{qual}\n{doc}\n{body}".strip()[:_MAX]
                yield CodeUnit(id=f"{repo_id}:{relpath}:{qual}", text=text or qual, path=relpath,
                               language=lang, symbol_type=sym, qualname=qual)
    return _TS()


def register_treesitter(register_extractor, CodeUnit, *, record_error: Callable[[str, str], None] | None = None) -> set:
    """Register a tree-sitter extractor for every grammar whose wheel is installed. No-op + returns an
    empty set if ``tree_sitter`` itself is missing (codna then stays Python-only). Returns the live set."""
    try:
        import tree_sitter  # noqa: F401
    except Exception:
        return set()
    by_lang: dict[str, list] = {}
    for ext, lang in EXT_LANG.items():
        by_lang.setdefault(lang, []).append(ext)
    live = set()
    for lang, exts in by_lang.items():
        mod, _acc = GRAMMAR_MOD[lang]
        if importlib.util.find_spec(mod) is None:
            continue
        try:
            _parser(lang)  # confirm the grammar wheel loads before registering
        except Exception as exc:  # noqa: BLE001 - installed grammar failed; record it for explicit requests.
            if record_error is not None:
                record_error(lang, f"{mod}: {type(exc).__name__}: {exc}")
            continue
        register_extractor(_make_extractor(CodeUnit, lang, exts))
        live.add(lang)
    return live
