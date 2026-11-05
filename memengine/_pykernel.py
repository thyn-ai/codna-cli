"""ctypes-shaped adapter that answers the kernel's C ABI with pure NumPy.

The engine calls the native kernel through a handful of `ctypes.CDLL` attributes whose signatures were bound
once in :func:`memengine.mojo_backend._lib`. Every call site passes CALLER-OWNED buffers as ``c_void_p``
(``arr.ctypes.data_as(ctypes.c_void_p)``) plus ``c_int32``/``c_float`` scalars, and reads results back out of
those same buffers.

This module supplies an object with exactly those attributes, implemented by :mod:`memengine._pynative`. Because
it preserves the ABI shape -- pointer in, in-place write out, ``Int32`` return -- the call sites in
``partitioned.py``, ``lexical.py`` and ``embedding.py`` do not change at all. That is the point: the kernel stays
the fast path, and the fallback is a drop-in with no second code path to drift from the first.

WHAT IS AND IS NOT PRESERVED
----------------------------
* Integer-domain work (n-gram hashing, int8 quantisation rounding, BM25 token identity) is bit-identical.
* Float-domain work (inner products, BM25 score accumulation) agrees to float32 rounding because the kernel
  sums along SIMD lanes and NumPy sums pairwise. Ranking is identical on every input we have measured; scores
  can differ in the last few ULPs. Do not tighten a comparison here to exact equality -- that would be
  asserting something the hardware cannot promise, on either implementation.
* Speed is NOT preserved. The fallback is slower on large inputs; :mod:`memengine._pynative` states which
  parts are vectorised and which are interpreted, and ``cli/bench/bench_pynative_equivalence.py`` measures the
  gap on this machine.

The adapter is deliberately noisy, once: :func:`_pynative.KernelFallback` reports activation on stderr the
first time it is used, so "the engine is running without its native kernel" is a fact an operator can see
rather than infer. A silent substitution would make every performance claim about this engine unfalsifiable.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np

from memengine import _pynative as _pn

__all__ = ["PyKernel", "scalar"]


def scalar(value) -> float:
    """Unwrap a ``ctypes`` scalar (``c_float``/``c_int32``/...) to a Python number; pass plain numbers through.

    Call sites pass ``_F32(self.k1)`` in some places and a bare ``k`` in others, so the adapter must accept both.
    """
    if isinstance(value, ctypes._SimpleCData):  # noqa: SLF001 — the documented way to detect a ctypes scalar
        return value.value
    return value


def _f32_view(ptr, n: int) -> np.ndarray:
    """A float32 VIEW over ``ptr`` (no copy) of ``n`` elements. Writes through it land in the caller's buffer."""
    if n <= 0:
        return np.empty(0, np.float32)
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(int(n),))


def _f16_view(ptr, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, np.float16)
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16)), shape=(int(n),)).view(
        np.float16
    )


def _i8_view(ptr, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, np.int8)
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int8)), shape=(int(n),))


def _u8_view(ptr, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, np.uint8)
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(int(n),))


def _i32_view(ptr, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, np.int32)
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32)), shape=(int(n),))


def _i64_view(ptr, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, np.int64)
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int64)), shape=(int(n),))


def _bytes_view(ptr, n: int) -> bytes:
    if n <= 0:
        return b""
    return bytes(_u8_view(ptr, n).tobytes())


class _BoundKernelFn:
    """One ``ame_*`` entry point, shaped like a ``ctypes`` foreign function.

    The only two things a binding site does to a ``_FuncPtr`` are assign ``restype``/``argtypes`` and then call
    it. Both work here: the assignments are accepted and ignored (there is no libffi to configure), and calls
    forward to the NumPy implementation.
    """

    __slots__ = ("name", "impl", "restype", "argtypes")

    def __init__(self, name: str, impl) -> None:
        self.name = name
        self.impl = impl
        self.restype = ctypes.c_int32
        self.argtypes = None

    def __call__(self, *args, **kwargs):
        return self.impl(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<pure-NumPy kernel fn {self.name}>"


_warned = [False]


def _announce_once() -> None:
    """Report the degradation ONCE per process -- visible, never silent (see the module docstring)."""
    if _warned[0]:
        return
    _warned[0] = True
    print(
        "codna: the native AME kernel is not available on this platform; using the pure-NumPy reference "
        "implementation. Retrieval and scoring are unchanged (identical rankings, float32-identical scores); "
        "large vector/BM25 workloads are slower. Set TELYS_KERNEL to use the native kernel.",
        file=sys.stderr,
    )


class PyKernel:
    """Drop-in for the ``ctypes.CDLL`` object, with the kernel's C signatures answered in NumPy.

    Accessed like a library object -- ``k.ame_flat_topk`` -- so binding sites that also assign ``.restype`` and
    ``.argtypes`` keep working: attribute access returns a small wrapper that ignores those assignments (they
    configure libffi, which has no role here) and forwards the call to the matching ``_impl_*`` method.

    Methods write results into the caller's output buffers exactly where the native library would, so callers
    that read ``out_ids[j]``/``out_dist[j]`` up to a returned count keep working untouched.
    """

    def __init__(self) -> None:
        _announce_once()
        self._bound: dict[str, _BoundKernelFn] = {}

    def __getattr__(self, name: str):
        """Resolve ``ame_*`` to a bindable wrapper around ``_impl_ame_*``. Anything else is a real AttributeError."""
        if not name.startswith("ame_"):
            raise AttributeError(name)
        impl = getattr(self, f"_impl_{name}", None)
        if impl is None:
            raise AttributeError(f"the pure-NumPy reference does not implement {name}")
        bound = self._bound.get(name)
        if bound is None:
            bound = _BoundKernelFn(name, impl)
            self._bound[name] = bound
        return bound

    # ── exact / quantised flat top-k ────────────────────────────────────────────────────────────
    def _impl_ame_flat_top1(self, base, n, d, query) -> int:
        nn, dd = int(scalar(n)), int(scalar(d))
        best = -1
        if nn > 0:
            view = _f32_view(base, nn * dd).reshape(nn, dd)
            scores = view @ _f32_view(query, dd)
            best = int(np.argmax(scores))
        return int(best)

    def _impl_ame_flat_topk(self, base, n, d, query, k, out_ids, out_dist) -> int:
        nn, dd, kk = int(scalar(n)), int(scalar(d)), int(scalar(k))
        ids, sc = _pn.flat_topk(_f32_view(base, nn * dd).reshape(nn, dd), nn, dd, _f32_view(query, dd), kk)
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    def _impl_ame_flat_topk_f16(self, base_sorted, base_f16, n, d, query, k, cand_cap,
                          scratch_id, scratch_s, out_ids, out_dist) -> int:
        nn, dd, kk = int(scalar(n)), int(scalar(d)), int(scalar(k))
        cap = int(scalar(cand_cap))
        ids, sc = _pn.flat_topk_f16(
            _f32_view(base_sorted, nn * dd).reshape(nn, dd),
            _f16_view(base_f16, nn * dd).reshape(nn, dd),
            nn, dd, _f32_view(query, dd), kk, cap,
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    def _impl_ame_flat_topk_q8(self, base_sorted, base_q8, n, d, query, query_q8, k, cand_cap,
                         scratch_id, scratch_s, out_ids, out_dist) -> int:
        nn, dd, kk = int(scalar(n)), int(scalar(d)), int(scalar(k))
        cap = int(scalar(cand_cap))
        # The caller hands us a query_q8 scratch buffer the native kernel fills; fill it for symmetry so a caller
        # reading it back after the call sees the same contents either way.
        qq8 = _i8_view(query_q8, dd)
        _pn.quantize_i8(_f32_view(query, dd), 1, dd, qq8)
        ids, sc = _pn.flat_topk_q8(
            _f32_view(base_sorted, nn * dd).reshape(nn, dd),
            _i8_view(base_q8, nn * dd).reshape(nn, dd),
            nn, dd, _f32_view(query, dd), kk, cap,
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    def _impl_ame_flat_topk_i8(self, base_q8, n, d, query, query_q8, k, out_ids, out_dist) -> int:
        nn, dd, kk = int(scalar(n)), int(scalar(d)), int(scalar(k))
        if kk <= 0:
            return 0
        _pn.quantize_i8(_f32_view(query, dd), 1, dd, _i8_view(query_q8, dd))
        ids, sc = _pn.flat_topk_i8(
            _i8_view(base_q8, nn * dd).reshape(nn, dd), nn, dd, _f32_view(query, dd), kk
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    def _impl_ame_flat_topk_pq(self, codes, n, m, ksub, dsub, codebook, query, lut, k, out_ids, out_dist) -> int:
        nn, mm, ks, ds, kk = (int(scalar(x)) for x in (n, m, ksub, dsub, k))
        ids, sc = _pn.flat_topk_pq(
            _u8_view(codes, nn * mm).reshape(nn, mm), nn, mm, ks, ds,
            _f32_view(codebook, mm * ks * ds), _f32_view(query, ds * mm), kk,
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    # ── IVF ─────────────────────────────────────────────────────────────────────────────────────
    def _impl_ame_ivf_search(self, centroids, nlist, base_sorted, offsets, d, query, nprobe, k,
                       out_ids, out_dist) -> int:
        nl, dd, npb, kk = (int(scalar(x)) for x in (nlist, d, nprobe, k))
        total = _offset_total(offsets, nl)
        ids, sc = _pn.ivf_search(
            _f32_view(centroids, nl * dd).reshape(nl, dd), nl,
            _f32_view(base_sorted, total * dd), _i32_view(offsets, nl + 1),
            dd, _f32_view(query, dd), npb, kk,
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    def _impl_ame_ivf_search_f16(self, centroids, nlist, base_sorted, base_f16, offsets, d, query, nprobe, k,
                           cand_cap, scratch_id, scratch_s, out_ids, out_dist) -> int:
        nl, dd, npb, kk = (int(scalar(x)) for x in (nlist, d, nprobe, k))
        cap = int(scalar(cand_cap))
        total = _offset_total(offsets, nl)
        ids, sc = _pn.ivf_search_f16(
            _f32_view(centroids, nl * dd).reshape(nl, dd), nl,
            _f32_view(base_sorted, total * dd), _f16_view(base_f16, total * dd),
            _i32_view(offsets, nl + 1), dd, _f32_view(query, dd), npb, kk, cap,
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    def _impl_ame_ivf_search_q8(self, centroids, nlist, base_sorted, base_q8, offsets, d, query, nprobe, k,
                          cand_cap, query_q8, scratch_id, scratch_s, out_ids, out_dist) -> int:
        nl, dd, npb, kk = (int(scalar(x)) for x in (nlist, d, nprobe, k))
        cap = int(scalar(cand_cap))
        _pn.quantize_i8(_f32_view(query, dd), 1, dd, _i8_view(query_q8, dd))
        total = _offset_total(offsets, nl)
        ids, sc = _pn.ivf_search_q8(
            _f32_view(centroids, nl * dd).reshape(nl, dd), nl,
            _f32_view(base_sorted, total * dd), _i8_view(base_q8, total * dd),
            _i32_view(offsets, nl + 1), dd, _f32_view(query, dd), npb, kk, cap,
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    # ── quantisers ──────────────────────────────────────────────────────────────────────────────
    def _impl_ame_quantize_i8(self, src, n, d, dst) -> int:
        nn, dd = int(scalar(n)), int(scalar(d))
        _pn.quantize_i8(_f32_view(src, nn * dd), nn, dd, _i8_view(dst, nn * dd))
        return 0

    def _impl_ame_quantize_f16(self, src, n, d, dst) -> int:
        nn, dd = int(scalar(n)), int(scalar(d))
        _pn.quantize_f16(_f32_view(src, nn * dd), nn, dd, _f16_view(dst, nn * dd))
        return 0

    # ── coarse quantiser ────────────────────────────────────────────────────────────────────────
    def _impl_ame_kmeans_build(self, data, n, d, k, max_iter, out_centroids, out_labels) -> int:
        nn, dd, kk, mi = (int(scalar(x)) for x in (n, d, k, max_iter))
        cents, labels = _pn.kmeans_build(_f32_view(data, nn * dd).reshape(nn, dd), nn, dd, kk, mi)
        _f32_view(out_centroids, kk * dd)[:] = cents[:kk].reshape(-1)
        _i32_view(out_labels, nn)[:] = labels
        return 0

    # ── embedders ───────────────────────────────────────────────────────────────────────────────
    def _impl_ame_embed_bigram_dim(self) -> int:
        return _pn.bigram_dim()

    def _impl_ame_embed_unigram_dim(self) -> int:
        return _pn.unigram_dim()

    def _impl_ame_embed_trigram_dim(self) -> int:
        return _pn.trigram_dim()

    def _impl_ame_embed_multigram_dim(self) -> int:
        return _pn.multigram_dim()

    def _impl_ame_embed_bigram(self, text, n, out_vec) -> None:
        nn = int(scalar(n))
        _f32_view(out_vec, _pn.BIGRAM_DIM)[:] = _pn.bigram_embed(_bytes_view(text, nn), nn)
        return None

    def _impl_ame_embed_multigram(self, text, n, out_vec) -> None:
        nn = int(scalar(n))
        _f32_view(out_vec, _pn.MULTIGRAM_DIM)[:] = _pn.multigram_embed(_bytes_view(text, nn), nn)
        return None

    def _impl_ame_embed_neural(self, emb_table, emb_dim, vocab_rows, token_ids, count, out_vec) -> None:
        dd, rows, cc = int(scalar(emb_dim)), int(scalar(vocab_rows)), int(scalar(count))
        out = _pn.embed_neural(_f32_view(emb_table, rows * dd), dd, rows, _i32_view(token_ids, cc), cc)
        _f32_view(out_vec, dd)[:] = out
        return None

    # ── NSW ─────────────────────────────────────────────────────────────────────────────────────
    def _impl_ame_nsw_build(self, base, n, d, m, efc, neighbors, degree, visited, res_id, res_ip) -> int:
        nn, dd, mm, ef = (int(scalar(x)) for x in (n, d, m, efc))
        nb, deg = _pn.nsw_build(_f32_view(base, nn * dd).reshape(nn, dd), nn, dd, mm, ef)
        _i32_view(neighbors, nn * mm)[:] = nb.reshape(-1)
        _i32_view(degree, nn)[:] = deg
        # The kernel zeroes `visited` and leaves res_id/res_ip as scratch; mirror the zeroing so a caller that
        # reads them back sees a consistent state.
        _i32_view(visited, nn)[:] = 0
        return 0

    def _impl_ame_nsw_search(self, base, n, d, m, neighbors, degree, entry, query, ef, k, visited, stamp,
                       res_id, res_ip, out_ids, out_dist) -> int:
        nn, dd, mm, eff, kk = (int(scalar(x)) for x in (n, d, m, ef, k))
        ids, sc = _pn.nsw_search(
            _f32_view(base, nn * dd).reshape(nn, dd), nn, dd, mm,
            _i32_view(neighbors, nn * mm).reshape(nn, mm), _i32_view(degree, nn),
            int(scalar(entry)), _f32_view(query, dd), eff, kk,
        )
        self._write_topk(ids, sc, kk, out_ids, out_dist)
        return 0

    # ── code lexical (BM25) ─────────────────────────────────────────────────────────────────────
    def _impl_ame_code_tokenize(self, text, n, out, cap, stop_hashes, n_stop) -> int:
        """CNN-free port of the kernel tokenizer: fold, split, stopword-skip, lite-stem, FNV-1a-64."""
        nn, cp = int(scalar(n)), int(scalar(cap))
        ns = int(scalar(n_stop))
        stop = set(int(x) for x in _i64_view(stop_hashes, ns).tolist())
        toks = _pn.code_tokenize_hashes(_bytes_view(text, nn).decode("utf-8", "ignore"), stop)
        outv = _i64_view(out, cp)
        take = min(len(toks), cp)
        outv[:take] = toks[:take]
        return int(take)

    def _impl_ame_bm25_topk(self, corpus_hashes, offsets, n_docs, q_hashes, q_idf, n_q, k1, b, avgdl, k,
                      out_ids, out_scores) -> int:
        nd, nq, kk = int(scalar(n_docs)), int(scalar(n_q)), int(scalar(k))
        total = int(_i32_view(offsets, nd + 1)[nd]) if nd > 0 else 0
        ids, sc = _pn.bm25_topk(
            _i64_view(corpus_hashes, total), _i32_view(offsets, nd + 1), nd,
            _i64_view(q_hashes, nq), _f32_view(q_idf, nq), nq,
            scalar(k1), scalar(b), scalar(avgdl), kk,
        )
        return self._write_bm25(ids, sc, out_ids, out_scores)

    def _impl_ame_bm25_csr_topk(self, post_docs, post_tfs, q_starts, q_ends, q_idf, n_q, doc_denom, n_docs,
                          scores, k1, k, out_ids, out_scores) -> int:
        nd, nq, kk = int(scalar(n_docs)), int(scalar(n_q)), int(scalar(k))
        starts = _i32_view(q_starts, nq)
        # The postings are a flat buffer with no explicit length. Every query range must lie inside it, and the
        # caller built both, so the largest range end IS the buffer length -- using anything else would alias
        # past the allocation.
        total = 0
        if nq > 0:
            total = int(max(int(_i32_view(q_ends, nq).max()), 0))
        ids, sc = _pn.bm25_csr_topk(
            _i32_view(post_docs, total), _i32_view(post_tfs, total),
            starts, _i32_view(q_ends, nq), _f32_view(q_idf, nq), nq,
            _f32_view(doc_denom, nd), nd, scalar(k1), kk,
        )
        # `scores` is caller-owned scratch the native kernel accumulates into; zero it for symmetry so a caller
        # never observes stale contents from a previous query.
        _f32_view(scores, nd)[:] = 0.0
        return self._write_bm25(ids, sc, out_ids, out_scores)

    # ── shared output writers ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _write_topk(ids: np.ndarray, scores: np.ndarray, k: int, out_ids, out_dist) -> None:
        """Write ``k`` slots of (id, score) exactly as the native kernels do, padding with their sentinels.

        The native kernels fill ALL k slots on entry with (id=-1, score=-1e30) and only overwrite the ones they
        fill, so a caller reading the full window sees sentinels rather than stale memory. Matching that keeps
        callers that scan for ``id >= 0`` correct.
        """
        oi = _i32_view(out_ids, k)
        od = _f32_view(out_dist, k)
        oi[:] = -1
        od[:] = np.float32(-1.0e30)
        n = min(k, int(ids.size))
        if n > 0:
            oi[:n] = ids[:n].astype(np.int32)
            od[:n] = scores[:n].astype(np.float32)

    @staticmethod
    def _write_bm25(ids: np.ndarray, scores: np.ndarray, out_ids, out_scores) -> int:
        """BM25 kernels return a COUNT and write a dense descending prefix; nothing is sentinel-padded."""
        n = int(ids.size)
        if n > 0:
            _i32_view(out_ids, n)[:] = ids.astype(np.int32)
            _f32_view(out_scores, n)[:] = scores.astype(np.float32)
        return n


def _offset_total(offsets, nlist: int) -> int:
    """Total posting/row count implied by an IVF ``offsets`` array (its last entry)."""
    if nlist <= 0:
        return 0
    return int(_i32_view(offsets, nlist + 1)[nlist])
