"""Structural guardrails (A–Z test plan: MOD-FILESIZE).

Enforces the modularity constraint — every source file in the `codna` package stays under
the 1000-line ceiling, so modules are split before they sprawl. Wired into CI so the
security pipeline cannot silently grow a monolith.
"""
from __future__ import annotations

import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "codna"
MAX_LINES = 1000


def _py_files():
    return sorted(PKG.rglob("*.py"))


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.name)
def test_mod_filesize_under_ceiling(path: pathlib.Path):
    lines = path.read_text(encoding="utf-8").count("\n") + 1
    assert lines < MAX_LINES, f"{path.name} has {lines} lines (>= {MAX_LINES}); split it"
