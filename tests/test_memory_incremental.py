"""Incremental re-index: re-embed only changed symbols (Merkle-style local content-hash manifest).

index() hashes each symbol (salted by the embedder space_id) and skips the embed/upsert for symbols whose
content is unchanged since the last index — so re-indexing a large repo touches deltas, not the whole tree.
Needs the Telys kernel (local embedder); skips cleanly where it isn't installed (the offline CI lane).
"""
from __future__ import annotations

import os

import pytest


def _mem(monkeypatch, repo, db):
    monkeypatch.setenv("CODNA_MEMORY_EMBED", "local")
    try:
        import codna.memory as M
        M.CodeMemory(repo, db_path=db)          # constructs + validates the embedder (needs the kernel)
        return M
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"telys/kernel unavailable: {type(exc).__name__}: {exc}")


def _write_repo(repo):
    os.makedirs(repo, exist_ok=True)
    for i in range(3):
        with open(os.path.join(repo, f"m{i}.py"), "w") as f:
            f.write("\n".join(f"def func_{i}_{j}(x):\n    '''thing {i} {j}'''\n    return x + {j}" for j in range(6)))


def test_incremental_reindex_skips_unchanged(tmp_path, monkeypatch):
    repo, db = str(tmp_path / "repo"), str(tmp_path / "db")
    _write_repo(repo)
    M = _mem(monkeypatch, repo, db)

    r1 = M.CodeMemory(repo, db_path=db).index()
    assert r1["indexed"] > 0 and r1["unchanged"] == 0          # first index embeds everything

    r2 = M.CodeMemory(repo, db_path=db).index()
    assert r2["indexed"] == 0 and r2["unchanged"] == r1["indexed"]  # nothing changed -> zero re-embeds

    # add one symbol to one file
    with open(os.path.join(repo, "m1.py"), "a") as f:
        f.write("\ndef brand_new(x):\n    '''added'''\n    return x * 99\n")
    r3 = M.CodeMemory(repo, db_path=db).index()
    assert r3["indexed"] == 1                                   # only the new symbol is embedded

    # the manifest is local state — deleting it forces a full re-embed (safe fallback)
    os.remove(os.path.join(db, M.HASH_FILENAME))
    r4 = M.CodeMemory(repo, db_path=db).index()
    assert r4["indexed"] > 0 and r4["unchanged"] == 0


def test_recall_unaffected_by_incremental(tmp_path, monkeypatch):
    repo, db = str(tmp_path / "repo"), str(tmp_path / "db")
    _write_repo(repo)
    M = _mem(monkeypatch, repo, db)
    mem = M.CodeMemory(repo, db_path=db)
    mem.index()
    mem2 = M.CodeMemory(repo, db_path=db)
    mem2.index()        # incremental no-op
    res = mem2.recall("thing 2", final_k=3)
    assert len(res["symbols"]) > 0                              # recall still works after an incremental pass
