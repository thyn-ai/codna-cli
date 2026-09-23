"""Tests for the multi-language tree-sitter extractor.

Skipped cleanly when the tree-sitter grammar wheels aren't installed in a source/dev checkout. Release
base wheels install these grammars so normal `pip install codna` memory is not Python-only.
"""
from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_go")  # representative grammar wheel; skip the module if grammars absent

from codna.codeunits import CodeUnit, FILTER_COLUMNS  # noqa: E402
from codna.codeunits_treesitter import _iter_symbols, register_treesitter  # noqa: E402


def _has(syms, name):
    # symbols are (qualname, symbol_type, doc, body); match on the last qualname component
    return any(q == name or q.endswith("." + name) for q, *_ in syms)


def test_extracts_function_and_leading_doc_js():
    js = "/** Add one to the value */\nfunction addOne(x) { return x + 1; }\n"
    syms = _iter_symbols("javascript", js)
    assert _has(syms, "addOne")
    add = next(s for s in syms if s[0] == "addOne")
    assert "Add one" in add[2] and add[1] == "function"   # leading JSDoc captured, typed as a function


def test_extracts_across_languages():
    # one generic walk — comments (//, ///, #, /** */) and methods-in-types all via the same code path
    cases = {
        "go": ("// Sum returns a plus b\nfunc Sum(a, b int) int { return a + b }\n", "Sum"),
        "rust": ("/// Doubles the input number\nfn double(n: i32) -> i32 { n * 2 }\n", "double"),
        "ruby": ("# Greets the given name\ndef greet(name)\n  name\nend\n", "greet"),
        "java": ("class K {\n  /** computes a checksum */\n  int csum(int x){ return x; }\n}\n", "csum"),
        "php": ("<?php\n/** Slugifies a title */\nfunction slugify($t){ return $t; }\n", "slugify"),
    }
    for lang, (src, want) in cases.items():
        syms = _iter_symbols(lang, src)
        assert _has(syms, want), f"{lang}: {want!r} not extracted (got {[s[0] for s in syms]})"


def test_register_treesitter_wires_extractors_with_provenance():
    reg = {}

    def fake_register(ex):
        for e in ex.extensions:
            reg[e] = ex

    live = register_treesitter(fake_register, CodeUnit)
    assert {"go", "rust", "typescript"} <= live          # the grammar wheels loaded + registered
    assert ".go" in reg and ".rs" in reg and ".ts" in reg

    units = list(reg[".go"].extract("github.com/o/r", "pkg/sum.go",
                                    "// Sum returns a plus b\nfunc Sum(a int) int { return a }\n"))
    assert units, "expected at least one Go unit"
    u = next(x for x in units if x.qualname == "Sum")
    assert u.id == "github.com/o/r:pkg/sum.go:Sum" and u.language == "go" and u.path == "pkg/sum.go"
    # metadata is EXACTLY the collection's filter columns (+ scope_key) — Telys ingest never KeyErrors
    md = u.metadata("github.com/o/r", "scope")
    assert set(md) == set(FILTER_COLUMNS) | {"scope_key"} and md["language"] == "go"


def test_license_boilerplate_is_not_treated_as_a_doc():
    # SPDX/Copyright headers document nothing specific — must be filtered, not attached as the symbol doc
    src = ("// SPDX-License-Identifier: GPL-2.0\n// Copyright (c) 2020 Acme\n"
           "func Helper(x int) int { return x }\n")
    syms = _iter_symbols("go", src)
    h = next(s for s in syms if s[0] == "Helper")
    assert "SPDX" not in h[2] and "Copyright" not in h[2]
