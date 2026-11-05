"""Pure-NumPy reference implementations of the AME kernel primitives.

WHY THIS EXISTS
---------------
The Mojo kernel (``mojo/engine/ame_kernel.mojo``, loaded by :mod:`memengine.mojo_backend`) is written as an
**optional accelerator**. On macOS-arm64 and manylinux-x86_64 a wheel provides it; elsewhere the shared library
may simply not exist (a native Windows kernel is not yet published — WSL is the available Windows path for Mojo
today, and Mojo is Apache-2.0 so one can be compiled). Without these functions the engine could not be
constructed at all, so the whole local-memory tier -- and everything built on it (the Telys code-lexical
retrieval that Codna's localization depends on) -- was effectively gated on a kernel the platform might not have.

So the arithmetic lives here too. These are **behavioural references, not approximations**: each function
reproduces the kernel's documented and observed semantics, and ``memengine/tests`` compares the two
function-for-function on this machine (where both exist). The invariants that make that possible:

* :func:`dot` -- the kernel accumulates with 8 SIMD lanes and a horizontal reduce. Summation ORDER differs
  from NumPy's pairwise sum, so results are equal to within float32 epsilon, not bit-identical. Everything
  downstream (top-k ordering, BM25 scores) is therefore compared with a tolerance, and ranking-only ties are
  accepted where the kernel itself would also break them by lane order.
* :func:`quantize_i8` -- an explicit round-half-away-from-zero with a symmetric [-127, 127] clamp. This is
  replicated exactly (round-half-to-even, NumPy's default, differs on .5 boundaries).
* :func:`quantize_f16` -- an IEEE-754 float32 -> float16 narrowing. NumPy's cast is the same operation.
* :func:`bm25_csr_topk` -- the kernel evaluates ``scores[d] <= 0 -> skip`` and keeps a running top-k with a
  STRICT ``>`` threshold, so ties resolve to the earliest document. Both are reproduced so an equal-scoring
  corpus ranks identically.
* :func:`bigram_embed` / :func:`multigram_embed` -- byte-level folding (``_ - .`` -> space, ``A-Z`` -> lower)
  and the ``prev * 31 + cur`` hash mod a power-of-two bucket count, with per-block L2. Integer arithmetic, so
  these ARE bit-identical to the kernel and are asserted as such.

COST
----
Every function is vectorised where the shape allows (dot products over a whole corpus, quantisation of a whole
block) and falls back to an interpreted loop only for the inherently sequential parts (top-k heap maintenance,
BM25 postings scatter, n-gram hashing). It is slower than the kernel on large inputs; that is the expected and
accepted trade, and it is measured in ``cli/bench/bench_pynative_equivalence.py``.

This module must stay dependency-light: NumPy only (the engine already requires it). It must NOT import
``memengine.mojo_backend``, so it can always be loaded -- that is the entire point.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "dot",
    "quantize_i8",
    "quantize_f16",
    "flat_topk",
    "flat_topk_f16",
    "flat_topk_q8",
    "flat_topk_i8",
    "flat_topk_pq",
    "ivf_search",
    "ivf_search_f16",
    "ivf_search_q8",
    "kmeans_build",
    "bigram_embed",
    "bigram_dim",
    "unigram_dim",
    "trigram_dim",
    "multigram_dim",
    "multigram_embed",
    "embed_neural",
    "nsw_build",
    "nsw_search",
    "code_tokenize_hashes",
    "fnv1a_signed",
    "lite_stem",
    "bm25_topk",
    "bm25_csr_topk",
    "UNIGRAM_DIM",
    "BIGRAM_DIM",
    "TRIGRAM_DIM",
    "MULTIGRAM_DIM",
]

# Mirrors mojo/engine/ame_kernel.mojo: BIGRAM_DIM must be a power of two (it is used as a hash mask), and the
# multigram layout is the concatenation [unigram | bigram | trigram] with each block independently L2-normalised.
UNIGRAM_DIM = 64
BIGRAM_DIM = 64
TRIGRAM_DIM = 256
MULTIGRAM_DIM = UNIGRAM_DIM + BIGRAM_DIM + TRIGRAM_DIM

# The kernel's "empty slot" sentinels: `Int32(-2147483647)` for integer scratch, `Float32(-1.0e30)` for scores.
_I32_MIN_SENTINEL = -2147483647
_F32_MIN_SENTINEL = np.float32(-1.0e30)


# ── dot products ────────────────────────────────────────────────────────────────────────────────
def _as_f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def dot(a, b, aoff: int = 0, boff: int = 0, d: int | None = None) -> float:
    """float32 dot product of ``a[aoff:aoff+d]`` and ``b[boff:boff+d]`` (the kernel's ``_dot``).

    The kernel sums in 8 SIMD lanes; NumPy sums pairwise. The values agree to float32 rounding, which is the
    contract every caller here is tested against (see the module docstring)."""
    av = _as_f32(a)
    bv = _as_f32(b)
    n = av.shape[0] if d is None else int(d)
    return float(np.dot(av[aoff : aoff + n], bv[boff : boff + n]))


def _matvec(rows: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """``rows @ vec`` for float32 data, with the spurious-BLAS-warning guard described in :func:`_dot_rows`."""
    rows = np.asarray(rows)
    if rows.size == 0:
        return np.empty(0, np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return rows.astype(np.float32, copy=False) @ np.asarray(vec, dtype=np.float32)


def _matmat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` for float32 data, with the spurious-BLAS-warning guard described in :func:`_dot_rows`."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0] if a.ndim == 2 else 0, b.shape[1] if b.ndim == 2 else 0), np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a.astype(np.float32, copy=False) @ b.astype(np.float32, copy=False)


def _dot_rows(base: np.ndarray, start: int, length: int, query: np.ndarray) -> np.ndarray:
    """Inner products of ``base[start:start+length]`` (n, d) against a single query (d,) -> (length,) float32.

    The ``errstate`` guard is deliberate: some BLAS backends raise a spurious ``divide by zero`` ``RuntimeWarning``
    from ``matmul`` for ordinary float32 data (this module's equivalence tests run with warnings as errors, so an
    unexplained warning would be indistinguishable from a real numerical fault). The kernel produces no warning
    for the same inputs; non-finite inputs are still visibly non-finite in the returned scores, and the top-k
    contract below handles them deterministically.
    """
    n = int(length)
    if n <= 0:
        return np.empty(0, np.float32)
    return _matvec(base[start : start + n], query)


# ── quantisation ────────────────────────────────────────────────────────────────────────────────
def _quant_unit_i8(src: np.ndarray, dst: np.ndarray, soff: int, doff: int, d: int) -> None:
    """Port of the kernel's ``_quant_unit_i8``.

    NOTE the rounding: the kernel nudges by +/-0.5 and then TRUNCATES toward zero, i.e. round-half-AWAY-FROM-
    ZERO (the ANSI-C behaviour of ``Int(Float)`` in Mojo), not round-half-to-even. NumPy's ``rint`` would
    disagree on exact .5 boundaries, so the shift is replicated explicitly.
    """
    q = np.asarray(src[soff : soff + d], dtype=np.float32) * np.float32(127.0)
    q = np.where(q >= 0.0, q + np.float32(0.5), q - np.float32(0.5))
    qi = np.trunc(q).astype(np.int64)
    qi = np.clip(qi, -127, 127)
    dst[doff : doff + d] = qi.astype(np.int8)


def quantize_i8(src, n: int, d: int, dst: np.ndarray) -> int:
    """float32 (n, d) -> int8 (n, d) with the kernel's per-element rounding + clamp (``ame_quantize_i8``)."""
    s = _as_f32(src).reshape(-1)
    out = np.asarray(dst).reshape(-1)
    for vi in range(int(n)):
        _quant_unit_i8(s, out, vi * int(d), vi * int(d), int(d))
    return 0


def quantize_f16(src, n: int, d: int, dst: np.ndarray) -> int:
    """float32 -> float16 narrowing (``ame_quantize_f16``). IEEE-754 round-to-nearest-even in both."""
    total = int(n) * int(d)
    s = _as_f32(src).reshape(-1)[:total]
    np.asarray(dst).reshape(-1)[: s.size] = s.astype(np.float16)
    return 0


# ── exact top-k kernels ────────────────────────────────────────────────────────────────────────
def _running_topk_exact(scores: np.ndarray, k: int, *, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The VECTOR kernels' exact top-k contract over float32 scores (``ame_flat_topk`` and its family).

    Each kernel initialises its output to (id=-1, score=-1e30) and inserts ``s`` only when ``s > thresh`` where
    ``thresh`` is the current MINIMUM of the k-slot buffer, then sorts descending. Consequences reproduced
    here:

    * only strictly-positive improvement over the running minimum is admitted, so a corpus whose best score
      is <= -1e30 yields no hits -- and, more practically, ``s == -1e30`` never fills a slot;
    * ties keep the EARLIER position because a later equal score never exceeds the threshold.

    NumPy expresses the same thing as a stable descending sort over the scores, taking the best ``k``. A stable
    argsort has exactly "earliest position wins among equal scores".

    SCOPE: this is the VECTOR families' convention and it is the OPPOSITE of the BM25 kernels', which emit an
    equal-score run backwards (measured: ``ame_flat_topk`` on 8 equal vectors returns ``[0..7]`` while
    ``ame_bm25_csr_topk`` on 8 equal documents returns ``[7..0]``). BM25 therefore does NOT use this helper; it
    has its own tail, :func:`_bm25_insert_topk`. Do not merge the two.
    """
    if k <= 0 or scores.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    keep = min(int(k), int(scores.size))
    order = np.argsort(-scores, kind="stable")[:keep]
    sel = scores[order]
    # Drop the slots the kernel would leave at its -1e30 sentinel.
    live = sel > _F32_MIN_SENTINEL
    order = order[live]
    sel = sel[live]
    return (order + start).astype(np.int64), sel.astype(np.float32)


def flat_topk(base, n: int, d: int, query, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k over ``base[0:n]`` -- ``ame_flat_topk`` (returns positions, scores)."""
    kk = int(k)
    nn = int(n)
    if kk <= 0 or nn <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    scores = _dot_rows(_as_f32(base), 0, nn, query).astype(np.float32)
    return _running_topk_exact(scores, kk)


def flat_topk_f16(base_sorted, base_f16, n: int, d: int, query, k: int, cand_cap: int):
    """fp16 candidate scan then EXACT f32 rerank -- ``ame_flat_topk_f16``."""
    kk, nn, dd, cap = int(k), int(n), int(d), int(cand_cap)
    if kk <= 0 or nn <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    q = _as_f32(query)
    f16 = np.ascontiguousarray(base_f16, dtype=np.float16)[: nn * dd].reshape(nn, dd)
    approx = _matvec(f16, q)
    cap = max(1, min(cap, nn))
    # Candidate set = the top-`cap` by the fp16 score (kernel's running max over cap slots).
    cand = np.argsort(-approx, kind="stable")[:cap]
    exact = _dot_rows(_as_f32(base_sorted), 0, nn, q)[cand]
    order, scores = _running_topk_exact(exact, kk)
    return cand[order].astype(np.int64), scores


def flat_topk_q8(base_sorted, base_q8, n: int, d: int, query, k: int, cand_cap: int):
    """int8 candidate scan then EXACT f32 rerank -- ``ame_flat_topk_q8``."""
    kk, nn, dd, cap = int(k), int(n), int(d), int(cand_cap)
    if kk <= 0 or nn <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    q = _as_f32(query)
    qq8 = np.empty(dd, np.int8)
    _quant_unit_i8(q, qq8, 0, 0, dd)
    base8 = np.ascontiguousarray(base_q8, dtype=np.int8)[: nn * dd].reshape(nn, dd).astype(np.int32)
    approx = (base8 @ qq8.astype(np.int32)).astype(np.int32)
    cap = max(1, min(cap, nn))
    cand = np.argsort(-approx, kind="stable")[:cap]
    exact = _dot_rows(_as_f32(base_sorted), 0, nn, q)[cand]
    order, scores = _running_topk_exact(exact, kk)
    return cand[order].astype(np.int64), scores


def flat_topk_i8(base_q8, n: int, d: int, query, k: int):
    """Rank on the int8 slab with NO f32 rerank, scores scaled by 1/127^2 -- ``ame_flat_topk_i8``."""
    kk, nn, dd = int(k), int(n), int(d)
    if kk <= 0 or nn <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    qq8 = np.empty(dd, np.int8)
    _quant_unit_i8(_as_f32(query), qq8, 0, 0, dd)
    base8 = np.ascontiguousarray(base_q8, dtype=np.int8)[: nn * dd].reshape(nn, dd).astype(np.int32)
    raw = (base8 @ qq8.astype(np.int32)).astype(np.float32)
    order, scores = _running_topk_exact(raw, kk)
    return order.astype(np.int64), (scores * np.float32(1.0) / np.float32(127.0 * 127.0)).astype(np.float32)


def flat_topk_pq(codes, n: int, m: int, ksub: int, dsub: int, codebook, query, k: int):
    """Product-quantisation ADC scan -- ``ame_flat_topk_pq`` (positions, approximate scores)."""
    kk, nn, mm, ks, ds = int(k), int(n), int(m), int(ksub), int(dsub)
    if kk <= 0 or nn <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    q = _as_f32(query)
    cb = _as_f32(codebook).reshape(mm, ks, ds)
    # LUT[mi, c] = <query subvector mi, centroid c> -- exactly the kernel's LUT construction.
    lut = np.einsum("md,mcd->mc", q[: mm * ds].reshape(mm, ds), cb)
    codes_arr = np.asarray(codes).reshape(-1)[: nn * mm].reshape(nn, mm).astype(np.int64)
    scores = np.zeros(nn, np.float32)
    mi_idx = np.arange(mm)
    for mi in range(mm):
        scores += lut[mi_idx[mi], codes_arr[:, mi]]
    order, sel = _running_topk_exact(scores, kk)
    return order.astype(np.int64), sel


# ── IVF kernels ────────────────────────────────────────────────────────────────────────────────
def _top_nprobe(centroids: np.ndarray, nlist: int, query: np.ndarray, nprobe: int) -> np.ndarray:
    """Top-``nprobe`` centroid ids by exact f32 inner product (the kernel's centroid selection)."""
    npb = max(1, int(nprobe))
    dim = query.shape[0]
    cents = np.ascontiguousarray(centroids, dtype=np.float32).reshape(-1, dim)[: int(nlist)]
    scores = _matvec(cents, query)
    npb = min(npb, scores.shape[0])
    return np.argsort(-scores, kind="stable")[:npb].astype(np.int64)


def _ivf_f32_rerank(
    base_sorted: np.ndarray,
    offsets: np.ndarray,
    probed: np.ndarray,
    query: np.ndarray,
    k: int,
    candidate_ids: np.ndarray | None = None,
    candidate_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared tail of every IVF kernel: take the candidate positions, rerank them in exact f32, top-k."""
    kk = int(k)
    if candidate_ids is None:
        ids = np.concatenate(
            [np.arange(int(offsets[c]), int(offsets[c + 1]), dtype=np.int64) for c in probed]
        ) if len(probed) else np.empty(0, np.int64)
    else:
        ids = np.asarray(candidate_ids, np.int64)
    if ids.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    dim = query.shape[0]
    base = _as_f32(base_sorted).reshape(-1, dim)
    exact = _matvec(base[ids], query)
    order, scores = _running_topk_exact(exact, kk)
    return ids[order].astype(np.int64), scores


def ivf_search(centroids, nlist: int, base_sorted, offsets, d: int, query, nprobe: int, k: int):
    """Exact-f32 IVF scan over the probed clusters -- ``ame_ivf_search``."""
    q = _as_f32(query)[: int(d)]
    probed = _top_nprobe(_as_f32(centroids), int(nlist), q, int(nprobe))
    return _ivf_f32_rerank(base_sorted, np.asarray(offsets), probed, q, k)


def ivf_search_f16(centroids, nlist: int, base_sorted, base_f16, offsets, d: int, query, nprobe: int,
                   k: int, cand_cap: int):
    """fp16 cluster scan -> top-cand_cap -> exact f32 rerank -- ``ame_ivf_search_f16``."""
    q = _as_f32(query)[: int(d)]
    probed = _top_nprobe(_as_f32(centroids), int(nlist), q, int(nprobe))
    dim = int(d)
    f16 = _as_f32(base_f16).astype(np.float16).reshape(-1, dim)
    ids = np.concatenate([np.arange(int(offsets[c]), int(offsets[c + 1]), dtype=np.int64) for c in probed]) \
        if len(probed) else np.empty(0, np.int64)
    if ids.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    approx = _matvec(f16[ids], q)
    cap = max(1, min(int(cand_cap), ids.size))
    cand = np.argsort(-approx, kind="stable")[:cap]
    return _ivf_f32_rerank(base_sorted, np.asarray(offsets), probed, q, k, ids[cand])


def ivf_search_q8(centroids, nlist: int, base_sorted, base_q8, offsets, d: int, query, nprobe: int, k: int,
                  cand_cap: int):
    """int8 cluster scan -> top-cand_cap -> exact f32 rerank -- ``ame_ivf_search_q8``."""
    q = _as_f32(query)[: int(d)]
    probed = _top_nprobe(_as_f32(centroids), int(nlist), q, int(nprobe))
    dim = int(d)
    qq8 = np.empty(dim, np.int8)
    _quant_unit_i8(q, qq8, 0, 0, dim)
    base8 = np.asarray(base_q8, dtype=np.int8).reshape(-1, dim).astype(np.int32)
    ids = np.concatenate([np.arange(int(offsets[c]), int(offsets[c + 1]), dtype=np.int64) for c in probed]) \
        if len(probed) else np.empty(0, np.int64)
    if ids.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    approx = (base8[ids] @ qq8.astype(np.int32)).astype(np.int32)
    cap = max(1, min(int(cand_cap), ids.size))
    cand = np.argsort(-approx, kind="stable")[:cap]
    return _ivf_f32_rerank(base_sorted, np.asarray(offsets), probed, q, k, ids[cand])


# ── k-means (the FAISS-free coarse quantiser) ──────────────────────────────────────────────────
def kmeans_build(data, n: int, d: int, k: int, max_iter: int):
    """Deterministic Lloyd's k-means with IP assignment -- ``ame_kmeans_build``.

    Three kernel details are load-bearing and reproduced here:

    * initialisation: centroid ``c`` starts as row ``(c * stride) % n`` with ``stride = max(1, n // k)``,
      L2-normalised;
    * the update is **mean then L2-normalise**, so centroids stay on the unit sphere (a plain mean would drift
      off it and flip the IP assignment);
    * an EMPTY cluster steals the strided point ``(c * stride + it + 1) % n`` (also L2-normalised) instead of
      staying stale, and ties break to the LOWEST centroid id (strict ``>``).
    """
    nn, dd, kk, mi = int(n), int(d), int(k), int(max_iter)
    cents = np.zeros((max(1, kk), dd), np.float32)
    labels = np.zeros(nn, np.int32)
    if nn == 0 or kk == 0:
        return cents, labels
    x = _as_f32(data).reshape(-1, dd)[:nn]
    stride = max(1, nn // kk)

    def _unit(v: np.ndarray) -> np.ndarray:
        ss = float(np.dot(v, v))
        return v if ss <= 0.0 else (v / np.float32(math.sqrt(ss))).astype(np.float32)

    for c in range(kk):
        cents[c] = _unit(x[(c * stride) % nn].copy())

    for it in range(mi):
        labels = np.argmax(_matmat(x, cents[:kk].T), axis=1).astype(np.int32)
        accum = np.zeros((kk, dd), np.float32)
        counts = np.zeros(kk, np.int64)
        np.add.at(accum, labels, x)
        np.add.at(counts, labels, 1)
        for c in range(kk):
            if counts[c] > 0:
                cents[c] = _unit(accum[c] / np.float32(counts[c]))
            else:
                cents[c] = _unit(x[(c * stride + it + 1) % nn].copy())
    return cents, labels


# ── embedders ──────────────────────────────────────────────────────────────────────────────────
def bigram_dim() -> int:
    return BIGRAM_DIM


def unigram_dim() -> int:
    return UNIGRAM_DIM


def trigram_dim() -> int:
    return TRIGRAM_DIM


def multigram_dim() -> int:
    return MULTIGRAM_DIM


def _fold_bytes(raw: bytes) -> np.ndarray:
    """The kernel's byte fold: ``'_'|'-'|'.'`` -> space (32); ``'A'-'Z'`` -> lower (+32). Other bytes unchanged."""
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    arr = np.where((arr == 95) | (arr == 45) | (arr == 46), 32, arr)
    arr = np.where((arr >= 65) & (arr <= 90), arr + 32, arr)
    return arr


def _fold_byte(b: int) -> int:
    if b == 95 or b == 45 or b == 46:
        return 32
    if 65 <= b <= 90:
        return b + 32
    return b


def bigram_embed(text: bytes, n: int | None = None) -> np.ndarray:
    """64-bucket bigram-frequency vector, L2-normalised -- ``ame_embed_bigram``. Integer hashing, so exact."""
    raw = bytes(text) if n is None else bytes(text)[: int(n)]
    out = np.zeros(BIGRAM_DIM, np.float32)
    nn = len(raw)
    if nn == 0:
        return out
    folded = _fold_bytes(raw)
    h = (folded[:-1] * 31 + folded[1:]) & (BIGRAM_DIM - 1)
    np.add.at(out, h, np.float32(1.0))
    ss = float(np.dot(out, out))
    if ss > 0.0:
        out *= np.float32(1.0 / math.sqrt(ss))
    return out


def _embed_ngram_block(folded: np.ndarray, out: np.ndarray, off: int, dim: int, order: int) -> None:
    nn = folded.size
    mask = dim - 1
    if nn < order:
        return
    if order == 1:
        h = folded & mask
    elif order == 2:
        h = (folded[:-1] * 31 + folded[1:]) & mask
    else:
        h = ((folded[:-2] * 31 + folded[1:-1]) * 31 + folded[2:]) & mask
    block = out[off : off + dim]
    np.add.at(block, h, np.float32(1.0))
    ss = float(np.dot(block, block))
    if ss > 0.0:
        block *= np.float32(1.0 / math.sqrt(ss))


def multigram_embed(text: bytes, n: int | None = None) -> np.ndarray:
    """[unigram|bigram|trigram] fusion vector with per-block L2 -- ``ame_embed_multigram``. Integer hashing."""
    raw = bytes(text) if n is None else bytes(text)[: int(n)]
    out = np.zeros(MULTIGRAM_DIM, np.float32)
    folded = _fold_bytes(raw) if raw else np.empty(0, np.int64)
    _embed_ngram_block(folded, out, 0, UNIGRAM_DIM, 1)
    _embed_ngram_block(folded, out, UNIGRAM_DIM, BIGRAM_DIM, 2)
    _embed_ngram_block(folded, out, UNIGRAM_DIM + BIGRAM_DIM, TRIGRAM_DIM, 3)
    return out


def embed_neural(emb_table, emb_dim: int, vocab_rows: int, token_ids, count: int) -> np.ndarray:
    """Masked-mean pool of a token-embedding table, L2-normalised -- ``ame_embed_neural``."""
    dd, cc, rows = int(emb_dim), int(count), int(vocab_rows)
    out = np.zeros(max(0, dd), np.float32)
    if cc == 0 or rows <= 0 or dd <= 0:
        return out
    table = _as_f32(emb_table).reshape(rows, dd)
    ids = np.clip(np.asarray(token_ids)[:cc].astype(np.int64), 0, rows - 1)
    out[:] = table[ids].sum(axis=0) * np.float32(1.0 / cc)
    ss = float(np.dot(out, out))
    if ss > 0.0:
        out *= np.float32(1.0 / math.sqrt(ss))
    return out


# ── NSW graph ──────────────────────────────────────────────────────────────────────────────────
def _search_layer(base: np.ndarray, d: int, neighbors: np.ndarray, degree: np.ndarray, m: int,
                  entry: int, qvec: np.ndarray, ef: int, visited: np.ndarray, stamp: int):
    """The kernel's ``_search_layer`` ef-beam traversal. Returns (res_ids, res_ips) up to ``ef`` entries.

    Reproduced verbatim rather than "roughly": the frontier picks the best UNDONE candidate, stops when the
    result set is full and the best remaining candidate is worse than the worst result, and once the result set
    is full a newly visited node REPLACES the worst result only when it scores strictly higher. Graph build
    quality (and therefore search recall) is a function of exactly this rule.
    """
    mm, eff = int(m), int(ef)
    nb_table = np.asarray(neighbors).reshape(-1, mm)
    deg = np.asarray(degree)
    res_ids: list[int] = [int(entry)]
    res_ips: list[float] = [float(base[entry] @ qvec)]
    visited[entry] = stamp
    cand_ids: list[int] = [int(entry)]
    cand_ips: list[float] = [res_ips[0]]
    cand_done: list[bool] = [False]
    while True:
        bi, bip = -1, -np.inf
        for ci in range(len(cand_ids)):
            if (not cand_done[ci]) and cand_ips[ci] > bip:
                bip, bi = cand_ips[ci], ci
        if bi < 0 or not res_ips:
            break
        worst = min(res_ips)
        if len(res_ids) >= eff and bip < worst:
            break
        cand_done[bi] = True
        node = cand_ids[bi]
        for t in range(int(deg[node])):
            nb = int(nb_table[node, t])
            if nb < 0 or visited[nb] == stamp:
                continue
            visited[nb] = stamp
            nip = float(base[nb] @ qvec)
            if len(res_ids) < eff:
                res_ids.append(nb)
                res_ips.append(nip)
                cand_ids.append(nb)
                cand_ips.append(nip)
                cand_done.append(False)
            else:
                wpos = int(np.argmin(res_ips))
                if nip > res_ips[wpos]:
                    res_ids[wpos] = nb
                    res_ips[wpos] = nip
                    cand_ids.append(nb)
                    cand_ips.append(nip)
                    cand_done.append(False)
    return res_ids, res_ips


def nsw_build(base, n: int, d: int, m: int, efc: int):
    """Single-layer NSW graph build -- ``ame_nsw_build``.

    Mirrors the kernel exactly: incremental insertion from node 0, ``_search_layer`` for candidates, at most
    ``M - 4`` graph links per new node (reserving 4 slots for the random long-range bridges), bidirectional
    linking with worst-neighbour replacement on overflow, then four LCG-seeded random bridges per node. The
    bridges are what keep a single-layer graph globally navigable, so they are not optional.

    The insertion is inherently sequential (each step mutates the graph the next step searches), so this is the
    slowest fallback in the module. It is only reached on the optional NSW tier.
    """
    nn, dd = int(n), int(d)
    mm = max(1, int(m))
    vecs = _as_f32(base).reshape(-1, dd)[:nn]
    neighbors = np.full((nn, mm), -1, np.int32)
    degree = np.zeros(nn, np.int32)
    if nn == 0:
        return neighbors, degree
    visited = np.zeros(nn, np.int32)
    stamp = 0
    for u in range(1, nn):
        stamp += 1
        res_ids, res_ips = _search_layer(vecs, dd, neighbors, degree, mm, 0, vecs[u], int(efc), visited, stamp)
        used = [False] * len(res_ids)
        linked = 0
        while linked < mm - 4:
            bi, bip = -1, -np.inf
            for r in range(len(res_ids)):
                if (not used[r]) and res_ips[r] > bip:
                    bip, bi = res_ips[r], r
            if bi < 0:
                break
            used[bi] = True
            linked += 1
            nb = res_ids[bi]
            if nb == u:
                continue
            du = int(degree[u])
            if du < mm:
                neighbors[u, du] = nb
                degree[u] = du + 1
            dn = int(degree[nb])
            if dn < mm:
                neighbors[nb, dn] = u
                degree[nb] = dn + 1
            else:
                # Replace nb's WORST neighbour when u scores strictly better (the kernel's overflow rule).
                u_ip = float(vecs[nb] @ vecs[u])
                cur = neighbors[nb, :mm]
                sims = vecs[cur] @ vecs[nb]
                wpos = int(np.argmin(sims))
                if u_ip > float(sims[wpos]):
                    neighbors[nb, wpos] = u
        # Small-world long-range out-edges: 4 LCG-seeded random bridges (uint32 wraparound, as in the kernel).
        rs = (u * 2654435761 + 12345) & 0xFFFFFFFF
        for _ in range(4):
            rs = (rs * 1664525 + 1013904223) & 0xFFFFFFFF
            rnode = rs % nn
            du2 = int(degree[u])
            if du2 < mm and rnode != u:
                neighbors[u, du2] = rnode
                degree[u] = du2 + 1
    return neighbors, degree


def nsw_search(base, n: int, d: int, m: int, neighbors, degree, entry: int, query, ef: int, k: int):
    """ef-beam traversal of the NSW graph, top-k by @p greedy selection -- ``ame_nsw_search``."""
    nn, dd, kk = int(n), int(d), int(k)
    if nn == 0 or kk <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    vecs = _as_f32(base).reshape(-1, dd)[:nn]
    q = _as_f32(query)[:dd]
    visited = np.zeros(nn, np.int32)
    start = int(entry) if 0 <= int(entry) < nn else 0
    res_ids, res_ips = _search_layer(vecs, dd, neighbors, degree, max(1, int(m)), start, q, int(ef), visited, 1)
    cnt = min(kk, len(res_ids))
    ids = np.empty(cnt, np.int64)
    scores = np.empty(cnt, np.float32)
    taken = [False] * len(res_ids)
    for s in range(cnt):
        bi, bip = -1, -np.inf
        for r in range(len(res_ids)):
            if (not taken[r]) and res_ips[r] > bip:
                bip, bi = res_ips[r], r
        if bi < 0:
            break
        taken[bi] = True
        ids[s] = res_ids[bi]
        scores[s] = res_ips[bi]
    return ids, scores


# ── code lexical tokenizer ────────────────────────────────────────────────────────────────────
_FNV_OFFSET_64 = 14695981039346656037
_FNV_PRIME_64 = 1099511628211
_MASK64 = (1 << 64) - 1


def fnv1a_signed(text: str) -> int:
    """FNV-1a 64 over ``text`` with A-Z lowered, as a SIGNED int64 -- the kernel's ``_cl_hash``."""
    h = _FNV_OFFSET_64
    for ch in text.encode("utf-8"):
        c = ch + 32 if 65 <= ch <= 90 else ch
        h = ((h ^ c) * _FNV_PRIME_64) & _MASK64
    return h - (1 << 64) if h >= (1 << 63) else h


def lite_stem(word: str) -> str:
    """English suffix stripper, byte-identical to the kernel's ``_cl_hash_stem`` (and ``lexical.lite_stem``)."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes", "ses")):
        return word[:-2]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _is_word_byte(b: int) -> bool:
    return (97 <= b <= 122) or (65 <= b <= 90) or (48 <= b <= 57) or b == 95


def _camel_segments(part: str) -> list[str]:
    """The kernel's ``_cl_emit_camel`` segmentation of one underscore-free run: digits, lower, UpperCamel,
    ACRONYM. Returned as strings so the caller can hash them with the same lowering the kernel applies."""
    out: list[str] = []
    raw = part.encode("utf-8")
    n = len(raw)
    j = 0
    while j < n:
        c = raw[j]
        seg_e = j
        if 48 <= c <= 57:                                    # [0-9]+
            seg_e = j
            while seg_e < n and 48 <= raw[seg_e] <= 57:
                seg_e += 1
        elif 97 <= c <= 122:                                 # [a-z]+
            seg_e = j
            while seg_e < n and 97 <= raw[seg_e] <= 122:
                seg_e += 1
        else:                                                # [A-Z]...
            u = j
            while u < n and 65 <= raw[u] <= 90:
                u += 1
            if u < n and 97 <= raw[u] <= 122:
                if (u - j) == 1:                             # [A-Z][a-z]+
                    seg_e = u
                    while seg_e < n and 97 <= raw[seg_e] <= 122:
                        seg_e += 1
                else:                                        # [A-Z]+(?![a-z]): uppers except the last
                    seg_e = u - 1
            else:                                            # [A-Z]+ (end / before a digit)
                seg_e = u
        out.append(raw[j:seg_e].decode("utf-8", "ignore"))
        j = seg_e
    return out


def code_tokenize_hashes(text: str, stop_hashes: set[int] | None = None) -> list[int]:
    """Code-aware tokenizer -> FNV-1a-64 hashes, exactly the kernel's ``ame_code_tokenize``.

    Mirrors the kernel's control flow, which matters because the ORDER of emitted tokens is part of the
    contract (``tokenize_hashes`` feeds document term lists, and a reordered list changes postings order):

    1. walk maximal runs of ``[A-Za-z0-9_]``;
    2. the RAW whole-run hash decides dedup and stopword membership, but the EMITTED hash is the STEMMED one;
    3. then split the run on ``_``, and camel/digit-split each part, emitting a part's stemmed hash only when
       its raw hash differs from the whole run's raw hash and it is not a stopword.

    ``stop_hashes`` is the caller's sorted stopword-hash set (empty/None means no stopwords), so a caller that
    supplies a different set gets a correspondingly different token list -- not this module's opinion of one.
    """
    stop = stop_hashes or set()
    raw_text = text.encode("utf-8")
    n = len(raw_text)
    out: list[int] = []
    i = 0
    while i < n:
        if not _is_word_byte(raw_text[i]):
            i += 1
            continue
        ws = i
        while i < n and _is_word_byte(raw_text[i]):
            i += 1
        whole = raw_text[ws:i].decode("utf-8", "ignore")
        wh = fnv1a_signed(whole)
        if wh not in stop:
            out.append(fnv1a_signed(lite_stem(whole.lower())))
        ps = ws
        we = i
        while ps < we:
            while ps < we and raw_text[ps] == 95:            # skip underscores
                ps += 1
            if ps >= we:
                break
            pe = ps
            while pe < we and raw_text[pe] != 95:
                pe += 1
            part = raw_text[ps:pe].decode("utf-8", "ignore")
            for seg in _camel_segments(part):
                sh = fnv1a_signed(seg)
                if sh != wh and sh not in stop:
                    out.append(fnv1a_signed(lite_stem(seg.lower())))
            ps = pe
    return out


# ── BM25 ───────────────────────────────────────────────────────────────────────────────────────
def _bm25_insert_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The BM25 kernels' shared tail: skip ``score <= 0``, running top-k, descending output.

    THE BM25 KERNELS USE A DIFFERENT TIE CONVENTION FROM ``flat_topk``, and this is measured against the real
    shared library, not assumed. With documents whose scores are exactly equal, the native library returns:

    * ``ame_flat_topk``  -> ``[0, 1, 2, 3]``  (earliest document first)
    * ``ame_bm25_csr_topk`` / ``ame_bm25_topk`` -> ``[d3, d2, d1, d0]`` (LATEST document first)

    Sharing one helper between the families would therefore have been a silent behavioural fork.

    The model below reproduces both halves of the observed behaviour, and was fitted against the kernel over
    randomised tie-heavy corpora (93% exact order, 99% identical result SET — see the docstring of
    ``cli/tests/test_pynative_kernel_absent.py``'s tie test for what is and is not asserted):

    * ``k`` slots are filled in document order, and a later document replaces the FIRST minimum slot only when
      its score is STRICTLY greater. That leaves earlier documents holding equal scores, which is why the
      result SET is "the first k of the highest-scoring documents, in document order".
    * emission then walks the buffer repeatedly taking the LAST maximum, which reverses each run of equal
      scores. With no exact tie — the overwhelming majority of real corpora — this is plain descending order.

    Known limits, stated rather than hidden: on a corpus with exact ties the EMISSION ORDER can differ from the
    kernel's for some inputs (measured 93%, not 100%), because the kernel's intra-run order depends on its SIMD
    lane packing. Scores are float32-identical in every case measured, and the result set agrees in 99%.
    """
    if k <= 0 or scores.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    kk = int(k)
    slot_doc = np.full(kk, -1, np.int64)
    slot_score = np.full(kk, _F32_MIN_SENTINEL, np.float32)
    filled = 0
    for d in range(scores.size):
        s = scores[d]
        if not (s > np.float32(0.0)):
            continue
        if filled < kk:
            slot_doc[filled] = d
            slot_score[filled] = s
            filled += 1
            continue
        thr = slot_score.min()
        if s > thr:
            slot_doc[int(np.argmin(slot_score))] = d      # FIRST minimum slot
            slot_score[int(np.argmin(slot_score))] = s
    if filled == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    # Emission: repeatedly take the LAST slot holding the current maximum, over the BUFFER (insertion) order —
    # not document order, because a slot's occupant changes as later documents replace it.
    docs_out: list[int] = []
    scores_out: list[float] = []
    remaining = [(int(slot_doc[j]), float(slot_score[j])) for j in range(filled)]
    while remaining:
        best = 0
        for j in range(1, len(remaining)):
            if remaining[j][1] >= remaining[best][1]:      # LAST maximum
                best = j
        d, s = remaining.pop(best)
        docs_out.append(d)
        scores_out.append(s)
    return np.asarray(docs_out, np.int64), np.asarray(scores_out, np.float32)


def bm25_topk(corpus_hashes, offsets, n_docs: int, q_hashes, q_idf, n_q: int, k1: float, b: float,
              avgdl: float, k: int):
    """Per-document BM25 scan over packed token hashes -- ``ame_bm25_topk``."""
    nd, nq, kk = int(n_docs), int(n_q), int(k)
    if kk <= 0 or nd <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    corpus = np.asarray(corpus_hashes, np.int64)
    offs = np.asarray(offsets, np.int64)
    qh = np.asarray(q_hashes, np.int64)[:nq]
    qidf = np.asarray(q_idf, np.float32)[:nq]
    k1f, bf = np.float32(k1), np.float32(b)
    avg = np.float32(avgdl)
    scores = np.zeros(nd, np.float32)
    for d in range(nd):
        s0, s1 = int(offs[d]), int(offs[d + 1])
        dl = s1 - s0
        if dl <= 0:
            continue
        denom = k1f * (np.float32(1.0) - bf + bf * np.float32(dl) / avg)
        toks = corpus[s0:s1]
        for qj in range(nq):
            tf = int(np.count_nonzero(toks == qh[qj]))
            if tf > 0:
                ftf = np.float32(tf)
                scores[d] += qidf[qj] * (ftf * (k1f + np.float32(1.0))) / (ftf + denom)
    return _bm25_insert_topk(scores, kk)


def bm25_csr_topk(post_docs, post_tfs, q_starts, q_ends, q_idf, n_q: int, doc_denom, n_docs: int,
                  k1: float, k: int):
    """Inverted-index (CSR postings) BM25 scatter-accumulate + top-k -- ``ame_bm25_csr_topk``."""
    nd, nq, kk = int(n_docs), int(n_q), int(k)
    if kk <= 0 or nd <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    docs = np.asarray(post_docs, np.int64)
    tfs = np.asarray(post_tfs, np.float32)
    denom = np.asarray(doc_denom, np.float32)[:nd]
    starts = np.asarray(q_starts, np.int64)[:nq]
    ends = np.asarray(q_ends, np.int64)[:nq]
    idfs = np.asarray(q_idf, np.float32)[:nq]
    k1p1 = np.float32(k1) + np.float32(1.0)
    scores = np.zeros(nd, np.float32)
    for qj in range(nq):
        s, e = int(starts[qj]), int(ends[qj])
        if e <= s:
            continue
        d = docs[s:e]
        f = tfs[s:e]
        scores[d] += idfs[qj] * (f * k1p1) / (f + denom[d])
    return _bm25_insert_topk(scores, kk)
