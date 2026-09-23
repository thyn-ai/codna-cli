"""Default semantic rerank = lexical-heavy BLEND (alpha*lexical + (1-alpha)*WordLlama, alpha=0.7).

The blend was the best-measured rerank (recall@10 0.775 vs 0.765 lexical-only / 0.665 WordLlama-only,
480-query CI study), so it's ON by default. Modes via CODNA_MEMORY_RERANK: blend (default) | wordllama
(pure) | lexical/off (none). Falls back to lexical if WordLlama is absent (never hard-fails). These tests
pin the gating + ordering with a fake model — no WordLlama package or native kernel needed.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")   # the reranker path uses numpy; skip cleanly in the minimal offline suite

import codna.memory as M  # noqa: E402


class _FakeWL:
    """Stand-in WordLlama: query + any text containing 'cookie' embed to [1,0]; everything else to [0,1]."""

    def embed(self, texts):
        return np.asarray([[1.0, 0.0] if (i == 0 or "cookie" in t) else [0.0, 1.0]
                           for i, t in enumerate(texts)], np.float32)


class _Mem:
    def __init__(self, texts):
        self._t = texts

    def _unit_texts(self, *a, **k):
        return self._t


# lexical scores prefer 'a'; WordLlama (cookie) prefers 'b'
_CANDS = [("a", 0.9), ("b", 0.5)]
_MEM = _Mem({"a": "def parse(): pass", "b": "def cookie_jar(): pass"})


def test_blend_is_default(monkeypatch):
    monkeypatch.delenv("CODNA_MEMORY_RERANK", raising=False)
    monkeypatch.setattr(M, "_wordllama", lambda: _FakeWL())
    out = M._semantic_rerank("lost cookies after redirect", _CANDS, _MEM)
    assert {i for i, _ in out} == {"a", "b"}                  # lossless: nothing dropped
    # default alpha=0.7 (lexical-heavy): lexical ('a') still wins, but WordLlama is mixed in (order may shift)
    assert out[0][0] == "a"


def test_alpha_extremes_select_each_signal(monkeypatch):
    monkeypatch.setattr(M, "_wordllama", lambda: _FakeWL())
    monkeypatch.setenv("CODNA_MEMORY_RERANK", "blend")
    monkeypatch.setenv("CODNA_MEMORY_RERANK_ALPHA", "1.0")    # pure lexical
    assert [i for i, _ in M._semantic_rerank("cookies", _CANDS, _MEM)] == ["a", "b"]
    monkeypatch.setenv("CODNA_MEMORY_RERANK_ALPHA", "0.0")    # pure WordLlama -> 'b' (cookie) wins
    assert M._semantic_rerank("cookies", _CANDS, _MEM)[0][0] == "b"


def test_lexical_off_disables(monkeypatch):
    monkeypatch.setattr(M, "_wordllama", lambda: (_ for _ in ()).throw(AssertionError("WL loaded while off")))
    for mode in ("off", "lexical", "none"):
        monkeypatch.setenv("CODNA_MEMORY_RERANK", mode)
        assert M._semantic_rerank("q", _CANDS, _MEM) == _CANDS    # no-op, WL never consulted


def test_pure_wordllama_mode(monkeypatch):
    monkeypatch.setenv("CODNA_MEMORY_RERANK", "wordllama")
    monkeypatch.setattr(M, "_wordllama", lambda: _FakeWL())
    out = M._semantic_rerank("lost cookies", _CANDS, _MEM)
    assert out[0][0] == "b"                                   # pure semantic promotes the cookie symbol


def test_default_blend_falls_back_to_lexical_when_wl_absent(monkeypatch):
    monkeypatch.delenv("CODNA_MEMORY_RERANK", raising=False)   # default blend
    monkeypatch.setattr(M, "_wordllama", lambda: (_ for _ in ()).throw(M.CodeMemoryError("no wordllama")))
    assert M._semantic_rerank("q", _CANDS, _MEM) == _CANDS     # graceful: returns lexical order, no raise


def test_explicit_wordllama_missing_raises(monkeypatch):
    monkeypatch.setenv("CODNA_MEMORY_RERANK", "wordllama")
    monkeypatch.setattr(M, "_wordllama", lambda: (_ for _ in ()).throw(M.CodeMemoryError("no wordllama")))
    with pytest.raises(M.CodeMemoryError):
        M._semantic_rerank("q", _CANDS, _MEM)


def test_passthrough_when_no_text(monkeypatch):
    monkeypatch.delenv("CODNA_MEMORY_RERANK", raising=False)
    monkeypatch.setattr(M, "_wordllama", lambda: (_ for _ in ()).throw(AssertionError("WL loaded with no text")))
    assert M._semantic_rerank("q", _CANDS, _Mem({})) == _CANDS   # no candidate text -> untouched
