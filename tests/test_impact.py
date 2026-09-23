"""Offline tests for codna.impact — repo-aware test-impact analysis (pure stdlib, no kernel).

Every test builds a synthetic repo under tmp_path: the engine must derive everything (package
aliases, test files, import edges) from the tree itself, so nothing here may assume a specific
repository layout.
"""
from __future__ import annotations

import os

from codna.impact import Impact, compute_impact


def _write(root, rel, text=""):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return rel


def _mini_repo(tmp_path):
    """src/pkg layout with one real package (`pkg`) and a pytest-style tests dir.

    pkg/core.py <- pkg/util.py <- tests/test_core.py (via the package-suffix alias `pkg.core`,
    which is exactly what src-layout tests import). Four tests total so a single affected test
    (25%) stays under the default 34% shared-core threshold."""
    _write(tmp_path, "src/pkg/__init__.py")
    _write(tmp_path, "src/pkg/core.py", "def core():\n    return 1\n")
    _write(tmp_path, "src/pkg/util.py", "from pkg.core import core\n\ndef util():\n    return core()\n")
    _write(tmp_path, "tests/test_core.py", "from pkg.util import util\n\ndef test_util():\n    assert util() == 1\n")
    _write(tmp_path, "tests/test_other.py", "def test_unrelated():\n    assert True\n")
    _write(tmp_path, "tests/test_a.py", "def test_a():\n    assert True\n")
    _write(tmp_path, "tests/test_b.py", "def test_b():\n    assert True\n")
    return tmp_path


class _Unit:
    def __init__(self, symbol_type):
        self.symbol_type = symbol_type


class _FakeCodna:
    def __init__(self, lang_by_ext, test_paths=()):
        self.lang_by_ext = lang_by_ext
        self.test_paths = set(test_paths)

    def language_for(self, rel):
        return self.lang_by_ext.get(os.path.splitext(rel)[1])

    def extract_file(self, repo_id, rel, source):
        return [_Unit("test" if rel in self.test_paths else "function")]


def test_python_change_reaches_importing_test_through_package_alias(tmp_path):
    root = _mini_repo(tmp_path)
    out = compute_impact(root, ["src/pkg/core.py"])
    assert out.mode == "subset"
    assert out.tests == ["tests/test_core.py"]   # via pkg.core alias -> util -> test_core


def test_changed_test_selects_only_itself(tmp_path):
    root = _mini_repo(tmp_path)
    out = compute_impact(root, ["tests/test_other.py"])
    assert out.mode == "subset"
    assert out.tests == ["tests/test_other.py"]


def test_unrelated_test_not_selected(tmp_path):
    root = _mini_repo(tmp_path)
    out = compute_impact(root, ["src/pkg/util.py"])
    assert out.mode == "subset"
    assert "tests/test_core.py" in out.tests
    assert "tests/test_other.py" not in out.tests


def test_broad_change_selects_full_suite(tmp_path):
    root = _mini_repo(tmp_path)
    _write(root, "pyproject.toml", "[project]\nname = 'x'\n")
    out = compute_impact(root, ["pyproject.toml"])
    assert out.mode == "all"


def test_non_python_change_without_codna_selects_full_suite(tmp_path):
    root = _mini_repo(tmp_path)
    _write(root, "web/app.ts", "export const x = 1;\n")
    out = compute_impact(root, ["web/app.ts"])
    assert out.mode == "all"
    assert "non-Python change" in out.reason


def test_disjoint_non_python_diff_selects_nothing(tmp_path):
    root = _mini_repo(tmp_path)
    _write(root, "web/app.ts", "export const x = 1;\n")
    _write(root, "web/util.tsx", "export const y = 2;\n")
    fake = _FakeCodna({".ts": "typescript", ".tsx": "tsx"})
    out = compute_impact(root, ["web/app.ts", "web/util.tsx"], codeunits=fake)
    assert out.mode == "subset" and out.tests == []
    assert "non-Python languages" in out.reason


def test_disjoint_falls_through_when_other_language_test_changed(tmp_path):
    root = _mini_repo(tmp_path)
    _write(root, "web/util.test.ts", "test('x', () => {});\n")
    fake = _FakeCodna({".ts": "typescript"}, test_paths={"web/util.test.ts"})
    out = compute_impact(root, ["web/util.test.ts"], codeunits=fake)
    assert out.mode == "all"


def test_disjoint_falls_through_for_unclassifiable_extension(tmp_path):
    root = _mini_repo(tmp_path)
    _write(root, "web/app.ts", "export const x = 1;\n")
    _write(root, "db/0001.sql", "SELECT 1;\n")
    fake = _FakeCodna({".ts": "typescript"})
    out = compute_impact(root, ["web/app.ts", "db/0001.sql"], codeunits=fake)
    assert out.mode == "all"


def test_shared_core_change_selects_full_suite(tmp_path):
    root = _mini_repo(tmp_path)
    # core reaches 1 of 4 tests = 25%; a stricter threshold flags it as shared core
    out = compute_impact(root, ["src/pkg/core.py"], shared_core_fraction=0.20)
    assert out.mode == "all"
    assert "shared core" in out.reason


def test_unparseable_changed_file_selects_full_suite(tmp_path):
    root = _mini_repo(tmp_path)
    _write(root, "src/pkg/broken.py", "def broken(:\n")
    out = compute_impact(root, ["src/pkg/broken.py"])
    assert out.mode == "all"


def test_relative_import_resolution(tmp_path):
    _write(tmp_path, "app/__init__.py")
    _write(tmp_path, "app/db.py", "def connect():\n    return True\n")
    _write(tmp_path, "app/service.py", "from .db import connect\n")
    _write(tmp_path, "tests/test_service.py", "from app.service import connect\n"
                                                "\ndef test_connect():\n    assert connect()\n")
    # filler tests keep the affected share (1/4) below the shared-core threshold
    _write(tmp_path, "tests/test_x.py", "def test_x():\n    assert True\n")
    _write(tmp_path, "tests/test_y.py", "def test_y():\n    assert True\n")
    _write(tmp_path, "tests/test_z.py", "def test_z():\n    assert True\n")
    out = compute_impact(tmp_path, ["app/db.py"])
    assert out.mode == "subset"
    assert out.tests == ["tests/test_service.py"]


def test_reexport_via_package_init_reaches_package_importers(tmp_path):
    """`from lib import fn` where lib/__init__.py re-exports fn from lib/thing.py: changing
    thing.py must reach the test, purely through resolved edges (no synthetic package climb)."""
    _write(tmp_path, "lib/__init__.py", "from lib.thing import fn\n")
    _write(tmp_path, "lib/thing.py", "def fn():\n    return 1\n")
    _write(tmp_path, "tests/test_lib.py", "from lib import fn\n\ndef test_fn():\n    assert fn() == 1\n")
    _write(tmp_path, "tests/test_p.py", "def test_p():\n    assert True\n")
    _write(tmp_path, "tests/test_q.py", "def test_q():\n    assert True\n")
    _write(tmp_path, "tests/test_r.py", "def test_r():\n    assert True\n")
    out = compute_impact(tmp_path, ["lib/thing.py"])
    assert out.mode == "subset"
    assert out.tests == ["tests/test_lib.py"]


def test_empty_change_list_selects_full_suite(tmp_path):
    root = _mini_repo(tmp_path)
    out = compute_impact(root, [])
    assert out.mode == "all"


def test_impact_dataclass_is_frozen():
    out = Impact("all", [], "x")
    try:
        out.mode = "subset"  # type: ignore[misc]
        assert False, "Impact must be immutable"
    except AttributeError:
        pass
