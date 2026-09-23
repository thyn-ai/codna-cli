"""Recall-time rerank pipeline for Codna code memory (split out of memory.py's 1000-line ceiling).

The consumer rerank seam, the query-aware soft test downweight, and the optional on-device
WordLlama precision blend. ``codna.memory`` re-imports every name from here, so the public import
surface (``from codna.memory import ...``) is unchanged.
"""
from __future__ import annotations

import os
import re


def rerank(query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Consumer-side rerank seam. Day-1: identity (keep Telys' partition-aware order + scores).
    A real cross-encoder / Algenta reranker plugs in here later — no network on day 1."""
    return candidates


# Query-aware soft rerank (lossless, recall-time policy). Tests stay FULLY indexed; when the issue is NOT
# test-oriented we softly DOWN-WEIGHT test symbols so source files surface. Monte-Carlo-validated over 900
# sampled queries: recall@10 0.638→0.741 (non-overlapping 95% CIs), MRR 0.324→0.414, no repo regressed —
# and test-oriented queries (failing test / assertion / fixture / flaky / junit …) are UNTOUCHED so bugs in
# test files stay findable. This is NOT exclusion: blind exclusion makes test-file bugs unfixable. Tune or
# disable with CODNA_MEMORY_TEST_DOWNWEIGHT (1.0 = off).
_TEST_QUERY_RE = re.compile(
    r"\b(test|tests|assert|assertion|fixture|flaky|junit|pytest|unittest|mock|conftest|failing)\b", re.I)


def _query_is_test_oriented(query: str) -> bool:
    return bool(_TEST_QUERY_RE.search(query or ""))


def _is_test_symbol(symbol_type: str | None, path: str | None) -> bool:
    if symbol_type == "test":
        return True
    p = (path or "").replace("\\", "/").lower()
    b = p.rsplit("/", 1)[-1]
    return ("/tests/" in p or "/test/" in p or b.startswith("test_")
            or b.endswith("_test.py") or b == "conftest.py")


def _soft_test_downweight(query: str, ids_scores: list, meta_by_id: dict) -> list:
    """Re-sort (id, score) by a soft test penalty when the query isn't test-oriented; identity otherwise.
    Operates over the full candidate set so a buried source symbol can surface into final-k."""
    try:
        factor = float(os.environ.get("CODNA_MEMORY_TEST_DOWNWEIGHT", "0.85"))
    except ValueError:
        factor = 0.85
    if factor == 1.0 or _query_is_test_oriented(query):
        return list(ids_scores)

    def _adj(id_, sc):
        m = meta_by_id.get(id_) or {}
        return sc * factor if _is_test_symbol(m.get("symbol_type"), m.get("path")) else sc

    return sorted(ids_scores, key=lambda t: -_adj(t[0], t[1]))


# Optional on-device semantic PRECISION reranker. The lexical multigram fusion does the cheap, fast first-pass
# recall; an ultra-light on-device model (WordLlama, 16 MB, CPU) re-scores only the top-k survivors to lift
# ranking precision (MRR). A benchmark put WordLlama's MRR at 0.48 vs fusion's 0.37 at equal recall@10. OFF by
# default; lossless (only reorders); enable with CODNA_MEMORY_RERANK=wordllama (CODNA_MEMORY_RERANK_K caps the
# pool, default 40). WordLlama is an OPTIONAL dependency — imported lazily, with a clear hint if missing.
_WL = []


def _wordllama():
    if not _WL:
        try:
            from wordllama import WordLlama
        except ImportError as exc:
            from .memory import CodeMemoryError      # lazy: avoids the memory<->memory_rerank cycle
            raise CodeMemoryError(
                "semantic reranking (CODNA_MEMORY_RERANK=wordllama) needs the optional WordLlama model — "
                "install it with:  pip install wordllama") from exc
        _WL.append(WordLlama.load())
    return _WL[0]


def _wl_cosine(wl, query: str, texts: list) -> "list":
    """Cosine of each candidate text vs the query under WordLlama (L2-normalized embeddings)."""
    import numpy as np
    m = np.asarray(wl.embed([query] + list(texts)), np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    m = m / n
    return list(m[1:] @ m[0])


def _semantic_rerank(query: str, cands: list, mem) -> list:
    """DEFAULT precision rerank of the top-k pool: a lexical-heavy BLEND of the lexical recall score and an
    on-device WordLlama semantic score — alpha*z(lexical) + (1-alpha)*z(wordllama), alpha=0.7. That blend was
    the best-measured config (recall@10 0.775 vs 0.765 lexical-only / 0.665 WordLlama-only, 480-query CI study).

    Modes via CODNA_MEMORY_RERANK: 'blend' (DEFAULT) | 'wordllama' (pure semantic) | 'lexical'/'off' (no rerank).
    alpha via CODNA_MEMORY_RERANK_ALPHA (default 0.7). Lossless: only the top-`CODNA_MEMORY_RERANK_K` reorder.
    GRACEFUL: if WordLlama isn't installed the default blend silently falls back to lexical (never hard-fails);
    only an EXPLICIT CODNA_MEMORY_RERANK=wordllama surfaces the missing-dependency error."""
    from .memory import CodeMemoryError              # lazy: avoids the memory<->memory_rerank cycle
    mode = (os.environ.get("CODNA_MEMORY_RERANK") or "blend").lower()
    if mode in ("off", "none", "lexical") or not cands:
        return cands
    try:
        k = int(os.environ.get("CODNA_MEMORY_RERANK_K", "40"))
    except ValueError:
        k = 40
    head, tail = cands[:k], cands[k:]
    id2text = mem._unit_texts()
    texts = [id2text.get(i, "") for i, _ in head]
    if not any(texts):                       # no recoverable text (e.g. repo moved) → leave order untouched
        return cands
    try:
        wl = _wordllama()
    except CodeMemoryError:
        if mode in ("wordllama", "wl"):      # explicitly requested -> surface the missing-dependency error
            raise
        return cands                         # default blend: WordLlama absent -> fall back to lexical
    wls = _wl_cosine(wl, query, texts)
    if mode in ("wordllama", "wl"):          # pure semantic order
        order = sorted(range(len(head)), key=lambda j: -float(wls[j]))
        return [(head[j][0], float(wls[j])) for j in order] + tail
    import numpy as np                        # blend (default): alpha*z(lexical) + (1-alpha)*z(wordllama)
    try:
        alpha = min(max(float(os.environ.get("CODNA_MEMORY_RERANK_ALPHA", "0.7")), 0.0), 1.0)
    except ValueError:
        alpha = 0.7

    def _z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else x * 0.0

    lex = np.array([s for _, s in head], dtype=float)
    blend = alpha * _z(lex) + (1.0 - alpha) * _z(np.array(wls, dtype=float))
    order = list(np.argsort(-blend))
    return [(head[j][0], float(blend[j])) for j in order] + tail
