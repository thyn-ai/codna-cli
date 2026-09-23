"""Offline tests for the structural code-unit extractor (pure stdlib — no Telys kernel)."""
from __future__ import annotations

import ast
import os
import sys

import pytest

from codna import codeunits as CU
from codna.codeunits import FILTER_COLUMNS, CodeUnit, PythonExtractor, UnsupportedLanguageError, extract_repo

MINI = os.path.join(os.path.dirname(__file__), "fixtures", "mini_repo")
REPO_ID = "test-repo"


def _force_python_only(monkeypatch):
    monkeypatch.setattr(CU, "_EXTRACTORS", {".py": PythonExtractor()})


def _by_id(repo_path=MINI):
    units, stats = extract_repo(repo_path, REPO_ID, languages=("python",))
    return {u.id: u for u in units}, stats


def test_extracts_expected_symbols_and_types():
    units, stats = _by_id()
    # module / class / method / function from refund.py + a test from test_refund.py
    assert units[f"{REPO_ID}:refund.py:RefundService"].symbol_type == "class"
    assert units[f"{REPO_ID}:refund.py:RefundService.process"].symbol_type == "method"
    assert units[f"{REPO_ID}:refund.py:reconcile"].symbol_type == "function"
    assert units[f"{REPO_ID}:refund.py:refund.py"].symbol_type == "module"   # module docstring unit
    assert units[f"{REPO_ID}:test_refund.py:test_reconciliation"].symbol_type == "test"


def test_text_carries_qualname_and_docstring():
    units, _ = _by_id()
    u = units[f"{REPO_ID}:refund.py:RefundService.process"]
    assert "RefundService.process" in u.text
    assert "Reconcile the ledger entry" in u.text   # docstring is embedded for retrieval


def test_metadata_keys_are_exactly_the_filter_columns():
    units, _ = _by_id()
    u = next(iter(units.values()))
    md = u.metadata(REPO_ID, "scope-x", commit_sha="abc", branch="main")
    assert set(md) == set(FILTER_COLUMNS) | {"scope_key"}   # never KeyErrors on ingest
    assert md["scope_key"] == "scope-x" and md["repo_id"] == REPO_ID


def test_syntaxerror_file_is_skipped_not_fatal():
    units, stats = _by_id()
    assert stats.skipped == 1                        # broken.py
    assert stats.files == 3                           # refund.py, test_refund.py, broken.py
    assert not any(":broken.py:" in i for i in units)  # nothing indexed from the bad file


def test_language_filter_excludes_non_python(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    (tmp_path / "notes.txt").write_text("def looks_like_code(): pass\n")
    units, stats = extract_repo(str(tmp_path), REPO_ID, languages=("python",))
    assert stats.files == 1 and any(":a.py:f" in u.id for u in units)


def test_language_filter_rejects_unregistered_language(tmp_path, monkeypatch):
    _force_python_only(monkeypatch)
    (tmp_path / "main.go").write_text("package main\nfunc main() {}\n")

    with pytest.raises(UnsupportedLanguageError) as excinfo:
        extract_repo(str(tmp_path), REPO_ID, languages=("go",))

    msg = str(excinfo.value)
    assert "unsupported language extractor(s): go" in msg
    assert "Registered languages: python" in msg
    assert "Reinstall `codna`" in msg
    assert "codna[memory-languages]" in msg


def test_language_filter_reports_recorded_registration_failure(monkeypatch):
    _force_python_only(monkeypatch)
    monkeypatch.setitem(CU._EXTRACTOR_ERRORS, "go", "tree_sitter_go: RuntimeError: broken parser")

    with pytest.raises(UnsupportedLanguageError) as excinfo:
        CU.resolve_languages(("go",))

    msg = str(excinfo.value)
    assert "unsupported language extractor(s): go" in msg
    assert "Registration failures: go: tree_sitter_go: RuntimeError: broken parser" in msg


def test_language_filter_reports_global_treesitter_registration_failure(monkeypatch):
    _force_python_only(monkeypatch)
    monkeypatch.setitem(CU._EXTRACTOR_ERRORS, "tree-sitter", "ImportError: bad package")

    with pytest.raises(UnsupportedLanguageError) as excinfo:
        CU.resolve_languages(("go",))

    msg = str(excinfo.value)
    assert "unsupported language extractor(s): go" in msg
    assert "Registration failures: tree-sitter: ImportError: bad package" in msg


def test_treesitter_registration_records_installed_grammar_failures(monkeypatch):
    from codna import codeunits_treesitter as TS

    monkeypatch.setitem(sys.modules, "tree_sitter", object())
    monkeypatch.setattr(
        TS.importlib.util,
        "find_spec",
        lambda name: object() if name == "tree_sitter_go" else None,
    )

    def broken_parser(lang):
        raise RuntimeError(f"{lang} parser failed")

    errors: dict[str, str] = {}
    monkeypatch.setattr(TS, "_parser", broken_parser)

    live = TS.register_treesitter(
        lambda _extractor: pytest.fail("broken grammar must not register an extractor"),
        CodeUnit,
        record_error=lambda lang, message: errors.setdefault(lang, message),
    )

    assert live == set()
    assert errors == {"go": "tree_sitter_go: RuntimeError: go parser failed"}


def test_default_language_filter_uses_all_registered_extractors(tmp_path, monkeypatch):
    class ToyExtractor:
        extensions = (".toy",)
        language = "toy"

        def extract(self, repo_id: str, relpath: str, source: str):
            yield CodeUnit(
                id=f"{repo_id}:{relpath}:toy_symbol",
                text=source,
                path=relpath,
                language=self.language,
                symbol_type="function",
                qualname="toy_symbol",
            )

    monkeypatch.setitem(CU._EXTRACTORS, ".toy", ToyExtractor())
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    (tmp_path / "b.toy").write_text("toy body\n")

    units, stats = extract_repo(str(tmp_path), REPO_ID)

    assert stats.files == 2
    assert any(u.language == "python" and u.path == "a.py" for u in units)
    assert any(u.language == "toy" and u.path == "b.toy" for u in units)


def test_ignores_the_memory_dir(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    mem = tmp_path / ".codna-memory"
    mem.mkdir()
    (mem / "leak.py").write_text("def should_not_be_indexed():\n    return 1\n")
    units, _ = extract_repo(str(tmp_path), REPO_ID, languages=("python",))
    assert not any("leak.py" in u.id for u in units)


def test_python_extractor_metadata():
    ex = PythonExtractor()
    assert ex.language == "python" and ".py" in ex.extensions


def test_language_for_maps_extensions_and_unknown(monkeypatch):
    _force_python_only(monkeypatch)
    assert CU.language_for("apps/a/b.py") == "python"
    assert CU.language_for("notes.md") is None
    assert CU.language_for("no_extension") is None

    class ToyExtractor:
        extensions = (".toy",)
        language = "toy"

        def extract(self, repo_id: str, relpath: str, source: str):
            return iter(())

    monkeypatch.setitem(CU._EXTRACTORS, ".toy", ToyExtractor())
    assert CU.language_for("x/y.toy") == "toy"


def test_extract_file_single_file_without_repo_walk(monkeypatch):
    _force_python_only(monkeypatch)
    units = CU.extract_file(REPO_ID, "svc/handler.py",
                            "def test_ok():\n    assert True\n\ndef helper():\n    return 1\n")
    kinds = {u.qualname: u.symbol_type for u in units}
    assert kinds["test_ok"] == "test" and kinds["helper"] == "function"
    assert all(u.path == "svc/handler.py" and u.language == "python" for u in units)


def test_extract_file_unknown_extension_and_parse_error(monkeypatch):
    _force_python_only(monkeypatch)
    assert CU.extract_file(REPO_ID, "README.md", "def not_code(): pass\n") == []
    with pytest.raises(SyntaxError):
        CU.extract_file(REPO_ID, "broken.py", "def broken(:\n")


# _segment must reproduce ast.get_source_segment byte-for-byte: the slice feeds both the embedding
# text and the incremental-index content hash, so ANY divergence would silently re-embed (or worse,
# mis-retrieve) every symbol. These sources stress the slicing rules: multibyte columns (byte vs
# char offsets), CRLF/CR line endings, form feeds, decorators, async, nested defs, blank/trailing.
_SEGMENT_CORPUS = (
    "def plain():\n    return 1\n",
    "def café():\n    return 'crème brûlée'\n",           # multibyte identifiers + literals
    "class A:\n    async def m(self,\n              y):\n        return y\n",
    "def decorated():\n    pass\ndecorated = staticmethod(decorated)\n",
    "@outer\n@inner\ndef stacked():\n    return 0\n",
    "def crlf():\r\n    return 1\r\n",                     # CRLF endings
    "def cr():\r    return 1\r",                           # bare CR endings
    "def feed():\n    x = 1\n\f\n    return x\n",          # form feed inside the body
    "def trailing():\n    return 1",                       # no trailing newline
    "\n\n\ndef late():\n    return 1\n",                   # leading blank lines
    "def tabbed():\n\treturn 1\n",                         # tab indentation
    "class Outer:\n    class Inner:\n        def deep(self):\n            return 'é'\n",
)


@pytest.mark.parametrize("source", _SEGMENT_CORPUS)
def test_segment_matches_stdlib_get_source_segment(source):
    lines = CU._LINE_SPLIT.findall(source)
    assert lines == ast._splitlines_no_ff(source)          # the split itself is faithful
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            expected = ast.get_source_segment(source, node) or ""
            assert CU._segment(lines, node) == expected, node.name


def test_extractor_text_unchanged_by_fast_segments():
    """End-to-end parity: every unit text equals what the old stdlib-segment flow produced."""
    src = "\n".join(_SEGMENT_CORPUS)
    fast_by_qual = {u.qualname: u.text for u in CU.extract_file(REPO_ID, "s.py", src)}

    # Reference walk replicating the pre-optimization flow (ast.get_source_segment per node).
    ref: dict[str, str] = {}

    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{node.name}"
                seg = ast.get_source_segment(src, node) or ""
                doc = ast.get_docstring(node) or ""
                ref[qual] = f"{qual}\n{doc}\n{seg}".strip()[:CU._MAX_UNIT_CHARS]
            elif isinstance(node, ast.ClassDef):
                qual = f"{prefix}{node.name}"
                seg = ast.get_source_segment(src, node) or ""
                first = seg.splitlines()[0] if seg else ""
                doc = ast.get_docstring(node) or ""
                ref[qual] = f"{qual}\n{first}\n{doc}".strip()[:CU._MAX_UNIT_CHARS]
                walk(node.body, f"{qual}.")

    walk(ast.parse(src).body, "")

    assert set(fast_by_qual) == set(ref)
    for qual, text in ref.items():
        assert fast_by_qual[qual] == text, qual
