"""Hybrid recall: dense + on-device BM25 (lexical) candidate union in CodeMemory.recall().

Dense vector search alone buries bare-identifier queries (cosine dilution: a 60-char qualname vs a
2000-char unit text landed ~rank 1900 on a real 54k-unit index); the lexical lane matches exact
tokens and restores them. Candidate-union semantics are tested offline with a fake collection; the
end-to-end lane (index builds it, recall consumes it, status reports it) needs the Telys kernel +
the lexical seam and skips cleanly where either is absent (older telys = dense-only, by design).
"""
from __future__ import annotations

import os

import pytest

from codna.memory import CodeMemory


class HybridFakeCollection:
    """search_text double with a dense lane and an optional lexical lane (mode='lexical')."""

    def __init__(self, dense, lexical=None, lexical_error=None) -> None:
        self.dense = dense                    # {"ids": [...], "scores": [...], "metadata": [...]}
        self.lexical = lexical
        self.lexical_error = lexical_error
        self.lexical_called = False

    def search_text(self, _query, *, top_k, where=None, explain=False, target_recall=None,
                    with_metadata=False, mode="dense"):
        if mode == "lexical":
            self.lexical_called = True
            if self.lexical_error is not None:
                raise self.lexical_error
            return self.lexical or {"ids": [], "scores": [], "metadata": []}
        res = dict(self.dense)
        if explain:
            res.setdefault("explain", {})
        return res

    def stats(self):
        return {"external_ids": len(self.dense.get("ids", []))}


def _mem(collection, lexical: bool) -> CodeMemory:
    mem = CodeMemory.__new__(CodeMemory)
    mem.repo_id = "repo"
    mem.service = None
    mem._scope_key = lambda repo_id, service, language: f"{repo_id}:{service}:{language}"
    mem._collection = lambda: collection
    mem._lexical = lexical
    # The WordLlama rerank seam pulls candidate texts via _unit_texts(); serve them from a cache so
    # the fake never touches a repo on disk.
    ids = list(collection.dense.get("ids", [])) + list((collection.lexical or {}).get("ids", []))
    mem.repo_path = "/nonexistent"
    mem._unit_text_cache = {i: i.split(":")[-1] for i in ids}
    return mem


def _dense(ids, metas):
    return {"ids": ids, "scores": [0.9 - 0.1 * i for i in range(len(ids))], "metadata": metas}


def test_hybrid_unions_lexical_only_candidates():
    dense = _dense(["a:src/a.py:f"], [{"path": "src/a.py", "symbol_type": "function"}])
    lexical = {"ids": ["b:src/b.py:exact_qualname_match"], "scores": [5.2],
               "metadata": [{"path": "src/b.py", "symbol_type": "function"}]}
    col = HybridFakeCollection(dense, lexical=lexical)
    mem = _mem(col, lexical=True)

    out = mem.recall("exact_qualname_match", final_k=8)

    assert col.lexical_called
    got = {s["id"] for s in out["symbols"]}
    assert "a:src/a.py:f" in got                      # dense candidate kept
    assert "b:src/b.py:exact_qualname_match" in got   # lexical-only candidate surfaced
    assert out["candidate_count"] == 2


def test_dense_only_when_lane_off():
    dense = _dense(["a:src/a.py:f"], [{"path": "src/a.py", "symbol_type": "function"}])
    col = HybridFakeCollection(dense, lexical={"ids": ["x"], "scores": [1.0], "metadata": [{}]})
    mem = _mem(col, lexical=False)

    out = mem.recall("anything")

    assert not col.lexical_called                     # no lexical call when the lane is off
    assert [s["id"] for s in out["symbols"]] == ["a:src/a.py:f"]


def test_lexical_lane_failure_falls_open_to_dense():
    dense = _dense(["a:src/a.py:f"], [{"path": "src/a.py", "symbol_type": "function"}])
    col = HybridFakeCollection(dense, lexical_error=RuntimeError("no lexical index"))
    mem = _mem(col, lexical=True)

    out = mem.recall("anything")

    assert col.lexical_called
    assert [s["id"] for s in out["symbols"]] == ["a:src/a.py:f"]   # dense result, no crash


def test_rrf_fusion_dedups_and_ranks_dual_lane_hits_first(monkeypatch):
    monkeypatch.setenv("CODNA_MEMORY_RERANK", "off")   # expose the raw fused order
    meta_b = [{"path": "src/b.py", "symbol_type": "function"}]
    dense = {"ids": ["x:src/x.py:x", "b:src/b.py:b"], "scores": [0.9, 0.8],
             "metadata": [{"path": "src/x.py", "symbol_type": "function"}] + meta_b}
    lexical = {"ids": ["b:src/b.py:b", "y:src/y.py:y"], "scores": [7.0, 3.0],
               "metadata": meta_b + [{"path": "src/y.py", "symbol_type": "function"}]}
    col = HybridFakeCollection(dense, lexical=lexical)
    mem = _mem(col, lexical=True)

    out = mem.recall("b", final_k=8)

    ids = [s["id"] for s in out["symbols"]]
    assert len(ids) == 3                              # union dedups the dual-lane hit
    # b ranks in BOTH lanes -> 1/62 + 1/61 beats either single-lane 1/61
    assert ids[0] == "b:src/b.py:b"


# ── end-to-end lane (needs the Telys kernel + the lexical seam) ─────────────────────────────────


def _mem_module(monkeypatch, repo, db):
    monkeypatch.setenv("CODNA_MEMORY_EMBED", "local")
    try:
        import codna.memory as M
        M.CodeMemory(repo, db_path=db)          # constructs + validates the embedder (needs the kernel)
        return M
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"telys/kernel unavailable: {type(exc).__name__}: {exc}")


def test_hybrid_lane_end_to_end(tmp_path, monkeypatch):
    repo, db = str(tmp_path / "repo"), str(tmp_path / "db")
    os.makedirs(repo, exist_ok=True)
    # A long, distinctive qualname in one file + distractor files with overlapping words: dense
    # cosine dilutes the bare-qualname query; the lexical lane must restore the exact match.
    with open(os.path.join(repo, "target.py"), "w") as f:
        f.write("def very_distinctive_zeta_connector_probe_handler(x):\n"
                "    '''zeta connector probe.'''\n    return x\n"
                + "\n".join(f"def filler_target_{i}(x):\n    return x + {i}" for i in range(12)))
    for i in range(6):
        with open(os.path.join(repo, f"distractor_{i}.py"), "w") as f:
            f.write("\n".join(
                f"def zeta_related_{i}_{j}(x):\n    '''connector distractor.'''\n    return x * {j}"
                for j in range(10)))

    M = _mem_module(monkeypatch, repo, db)
    if not M._sdk_supports_lexical(M.CodeMemory(repo, db_path=db)._eng):
        pytest.skip("telys SDK predates the lexical seam — dense-only by design")

    mem = M.CodeMemory(repo, db_path=db)
    mem.index()
    assert mem.status()["lexical"] is True            # lane built + reported

    out = mem.recall("very_distinctive_zeta_connector_probe_handler", final_k=8)
    assert out["symbols"], "hybrid recall returned nothing"
    assert out["symbols"][0]["path"] == "target.py"
    assert "very_distinctive_zeta_connector_probe_handler" in out["symbols"][0]["id"]

    # Reopen: the lane must survive persistence (tokens are stored; plex rebuilds on open).
    mem2 = M.CodeMemory(repo, db_path=db)
    out2 = mem2.recall("very_distinctive_zeta_connector_probe_handler", final_k=8)
    assert out2["symbols"][0]["path"] == "target.py"
