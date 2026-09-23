"""Extraction strips embedded NUL bytes from source (the on-device embedder rejects U+0000).

Some real repos (caveman, tailwindcss, grpc, Ventoy, TrafficMonitor) have source files with embedded NUL
bytes; without stripping, indexing them raised ValueError from the embedder. Pure-extraction test (no kernel).
"""
from __future__ import annotations

import codna.codeunits as CU


def test_extraction_strips_nul(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    # a valid-Python file carrying a NUL byte in a docstring
    (repo / "mod.py").write_text("def greet(name):\n    '''hi\x00 there'''\n    return 'hi ' + name\n")
    assert b"\x00" in (repo / "mod.py").read_bytes()       # the NUL is really on disk

    units, stats = CU.extract_repo(str(repo), "github.com/o/r", languages=("python",))
    assert stats.files == 1 and len(units) >= 1            # the file was scanned and a unit produced
    assert all("\x00" not in u.text for u in units)         # no unit text carries a NUL (embedder-safe)
    assert any("greet" in u.id for u in units)
