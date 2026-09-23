"""AME Mojo-native in-process backend (T0) — the REAL Mojo engine via C-ABI/ctypes.

Loads mojo_build/libame_kernel.dylib (built from mojo/engine/ame_kernel.mojo with
`mojo build --emit shared-lib`). Proves the Phase-0 FFI gate (GREEN): the in-process tier is the
ACTUAL Mojo SIMD kernel, not FAISS-via-Python.

Exact FLAT top-k: input + output buffers are caller-owned ctypes memory passed as
`UnsafePointer[T, MutExternalOrigin]` on the Mojo side (the routing_engine/cortex pattern) — outputs
are written in place, no returned-value/socket workaround.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

# Kernel resolution order: (1) TELYS_KERNEL / (2) AME_KERNEL env override; (3) a verified `telys runtime
# install` location ($TELYS_HOME/runtime/<platform>/, via telys.paths) — zero-config after an offline install;
# (4) a kernel bundled INSIDE the package at memengine/_runtime/ — present only in the INTERNAL full-engine
# wheel (TELYS_BUNDLE_RUNTIME=1 at build), never in the public SDK (D-30); (5) the repo-relative build path.
def _lib_ext() -> str:
    """Shared-library extension for this platform — mirrors telys.paths.lib_ext() but standalone,
    because memengine must stay dependency-free (it is what the telys SDK itself loads)."""
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform.startswith("win"):
        return "dll"
    return "so"


def _installed_kernel():
    try:
        from telys.paths import installed_lib_path
        return installed_lib_path("libame_kernel")
    except Exception:  # noqa: BLE001 — resolution must never break engine load
        return None


def _resolve_lib_path() -> str:
    p = os.environ.get("TELYS_KERNEL") or os.environ.get("AME_KERNEL")
    if p:
        return p
    installed = _installed_kernel()
    if installed:
        return installed
    ext = _lib_ext()
    here = os.path.dirname(__file__)
    bundled = os.path.join(here, "_runtime", f"libame_kernel.{ext}")
    if os.path.exists(bundled):
        return bundled
    return os.path.abspath(os.path.join(here, "..", "..", "..", "mojo_build", f"libame_kernel.{ext}"))


_LIB_PATH = _resolve_lib_path()
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        if not os.path.exists(_LIB_PATH):
            ext = _lib_ext()
            raise FileNotFoundError(
                f"Telys Mojo kernel not found at {_LIB_PATH}.\n"
                f"  • In a consumer repo: set TELYS_KERNEL=/abs/path/to/libame_kernel.{ext}\n"
                "  • To build it: cd mojo_env && pixi run mojo build --emit shared-lib "
                f"../mojo/engine/ame_kernel.mojo -o ../mojo_build/libame_kernel.{ext}"
            )
        lib = ctypes.CDLL(_LIB_PATH)
        lib.ame_flat_top1.restype = ctypes.c_int32
        lib.ame_flat_top1.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
        lib.ame_flat_topk.restype = ctypes.c_int32
        lib.ame_flat_topk.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.ame_ivf_search.restype = ctypes.c_int32
        lib.ame_ivf_search.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,                 # centroids, nlist
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,  # base_sorted, offsets, d
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,   # query, nprobe, k
            ctypes.c_void_p, ctypes.c_void_p,                 # out_ids, out_dist
        ]
        lib.ame_quantize_i8.restype = ctypes.c_int32
        lib.ame_quantize_i8.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,  # src, n, d, dst
        ]
        lib.ame_ivf_search_q8.restype = ctypes.c_int32
        lib.ame_ivf_search_q8.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,                   # centroids, nlist
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,  # base_sorted, base_q8, offsets, d
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,    # query, nprobe, k, cand_cap
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # query_q8, scratch_id, scratch_s
            ctypes.c_void_p, ctypes.c_void_p,                  # out_ids, out_dist
        ]
        lib.ame_kmeans_build.restype = ctypes.c_int32
        lib.ame_kmeans_build.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # data, n, d, k, max_iter
            ctypes.c_void_p, ctypes.c_void_p,                  # out_centroids, out_labels
        ]
        lib.ame_quantize_f16.restype = ctypes.c_int32
        lib.ame_quantize_f16.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
        lib.ame_ivf_search_f16.restype = ctypes.c_int32
        lib.ame_ivf_search_f16.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,                   # centroids, nlist
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,  # base_sorted, base_f16, offsets, d
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,    # query, nprobe, k, cand_cap
            ctypes.c_void_p, ctypes.c_void_p,                  # scratch_id, scratch_s
            ctypes.c_void_p, ctypes.c_void_p,                  # out_ids, out_dist
        ]
        lib.ame_flat_topk_f16.restype = ctypes.c_int32
        lib.ame_flat_topk_f16.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,   # base_sorted, base_f16, n, d
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,                    # query, k, cand_cap
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # scratch_id, scratch_s, out_ids, out_dist
        ]
        lib.ame_flat_topk_q8.restype = ctypes.c_int32
        lib.ame_flat_topk_q8.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,   # base_sorted, base_q8, n, d
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,   # query, query_q8, k, cand_cap
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # scratch_id, scratch_s, out_ids, out_dist
        ]
        # on-device embedders (pluggable providers wrap these): bigram (lexical) + pooled token embedder
        lib.ame_embed_bigram_dim.restype = ctypes.c_int32  # single source of truth for the bigram dim
        lib.ame_embed_bigram_dim.argtypes = []
        lib.ame_embed_bigram.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]  # text, n, out[dim]
        # compact-primary i8 scan: rank by int8 score with NO f32 rerank (so f32 need not be stored)
        lib.ame_flat_topk_i8.restype = ctypes.c_int32
        lib.ame_flat_topk_i8.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,   # base_q8, n, d
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,  # query(f32), query_q8(scratch), k
            ctypes.c_void_p, ctypes.c_void_p,                  # out_ids, out_dist
        ]
        # PQ ADC scan (32x tier): pure-Mojo asymmetric-distance scan over faiss-trained codes/codebook
        lib.ame_flat_topk_pq.restype = ctypes.c_int32
        lib.ame_flat_topk_pq.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # codes, n, m, ksub, dsub
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,  # codebook, query, lut(scratch), k
            ctypes.c_void_p, ctypes.c_void_p,                  # out_ids, out_dist
        ]
        # multigram fusion: concatenated [unigram | bigram | trigram], each block L2-normed (all dims from kernel)
        for _dimfn in ("ame_embed_unigram_dim", "ame_embed_trigram_dim", "ame_embed_multigram_dim"):
            getattr(lib, _dimfn).restype = ctypes.c_int32
            getattr(lib, _dimfn).argtypes = []
        lib.ame_embed_multigram.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]  # text, n, out[dim]
        lib.ame_embed_neural.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]  # table, emb_dim, vocab_rows, token_ids, count, out
        lib.ame_nsw_build.restype = ctypes.c_int32
        lib.ame_nsw_build.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # base,n,d,M,efc
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # neighbors, degree, visited
            ctypes.c_void_p, ctypes.c_void_p,                  # res_id, res_ip
        ]
        lib.ame_nsw_search.restype = ctypes.c_int32
        lib.ame_nsw_search.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # base,n,d,M
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,  # neighbors, degree, entry
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,   # query, ef, k
            ctypes.c_void_p, ctypes.c_int32,                   # visited, stamp
            ctypes.c_void_p, ctypes.c_void_p,                  # res_id, res_ip
            ctypes.c_void_p, ctypes.c_void_p,                  # out_ids, out_dist
        ]
        _LIB = lib
    return _LIB


class MojoFlatBackend:
    """In-process exact-FLAT top-k executed by the real Mojo SIMD kernel."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.lib = _lib()
        self.base: np.ndarray | None = None
        self.row_ids: np.ndarray | None = None
        self._bp = None
        self.n = 0

    def build(self, vectors: np.ndarray, row_ids: np.ndarray | None = None) -> None:
        self.base = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n = self.base.shape[0]
        self.row_ids = (
            np.asarray(row_ids, dtype=np.int64) if row_ids is not None
            else np.arange(self.n, dtype=np.int64)
        )
        self._bp = self.base.ctypes.data_as(ctypes.c_void_p)

    def search(self, queries: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        out = np.empty((q.shape[0], k), dtype=np.int64)
        oi = np.empty(k, dtype=np.int32)
        od = np.empty(k, dtype=np.float32)
        oip = oi.ctypes.data_as(ctypes.c_void_p)
        odp = od.ctypes.data_as(ctypes.c_void_p)
        scores = np.empty((q.shape[0], k), dtype=np.float32)
        for i in range(q.shape[0]):
            self.lib.ame_flat_topk(
                self._bp, self.n, self.dim, q[i].ctypes.data_as(ctypes.c_void_p), k, oip, odp
            )
            out[i] = self.row_ids[oi.astype(np.int64)]
            scores[i] = od
        return out, scores


class MojoIVFBackend:
    """In-process approximate IVF executed by the real Mojo SIMD kernel.

    One-time build (Python): k-means coarse quantizer (faiss.Kmeans) -> assign each vector to a
    centroid by IP -> group vectors contiguously by cluster. Hot path (Mojo `ame_ivf_search`):
    pick top-nprobe centroids, scan only those clusters' contiguous vectors with SIMD top-k.
    """

    def __init__(self, dim: int, nlist: int = 1024, nprobe: int = 32, kmeans: str = "faiss") -> None:
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.kmeans = kmeans  # "faiss" (BLAS, fast default) or "mojo" (FAISS-free, pure Mojo, slower)
        self.lib = _lib()

    def _train_centroids(self, base: np.ndarray) -> np.ndarray:
        n = base.shape[0]
        if self.kmeans == "mojo":  # pure-Mojo coarse quantizer (no FAISS dependency)
            cent = np.zeros((self.nlist, self.dim), dtype=np.float32)
            labels = np.zeros(n, dtype=np.int32)
            self.lib.ame_kmeans_build(
                base.ctypes.data_as(ctypes.c_void_p), n, self.dim, self.nlist, 10,
                cent.ctypes.data_as(ctypes.c_void_p), labels.ctypes.data_as(ctypes.c_void_p),
            )
            return np.ascontiguousarray(cent, dtype=np.float32)
        import faiss

        km = faiss.Kmeans(self.dim, self.nlist, niter=10, seed=42, verbose=False)
        km.train(base)
        return np.ascontiguousarray(km.centroids.reshape(self.nlist, self.dim), dtype=np.float32)

    def build(self, vectors: np.ndarray, row_ids: np.ndarray | None = None) -> None:
        base = np.ascontiguousarray(vectors, dtype=np.float32)
        n = base.shape[0]
        self.cent = self._train_centroids(base)
        assign = np.argmax(base @ self.cent.T, axis=1)  # nearest centroid by IP (matches the Mojo kernel)
        self.order = np.argsort(assign, kind="stable").astype(np.int64)  # sorted pos -> original index
        self.base_sorted = np.ascontiguousarray(base[self.order], dtype=np.float32)
        counts = np.bincount(assign, minlength=self.nlist)
        self.offsets = np.zeros(self.nlist + 1, dtype=np.int32)
        self.offsets[1:] = np.cumsum(counts).astype(np.int32)
        self.row_ids = (
            np.asarray(row_ids, dtype=np.int64) if row_ids is not None else np.arange(n, dtype=np.int64)
        )
        self._cp = self.cent.ctypes.data_as(ctypes.c_void_p)
        self._bsp = self.base_sorted.ctypes.data_as(ctypes.c_void_p)
        self._op = self.offsets.ctypes.data_as(ctypes.c_void_p)

    def search(self, queries: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        out = np.empty((q.shape[0], k), dtype=np.int64)
        scores = np.empty((q.shape[0], k), dtype=np.float32)
        oi = np.empty(k, dtype=np.int32)
        od = np.empty(k, dtype=np.float32)
        oip = oi.ctypes.data_as(ctypes.c_void_p)
        odp = od.ctypes.data_as(ctypes.c_void_p)
        for i in range(q.shape[0]):
            self.lib.ame_ivf_search(
                self._cp, self.nlist, self._bsp, self._op, self.dim,
                q[i].ctypes.data_as(ctypes.c_void_p), self.nprobe, k, oip, odp,
            )
            # out_ids are positions in base_sorted -> original index via order -> row id
            out[i] = self.row_ids[self.order[oi.astype(np.int64)]]
            scores[i] = od
        return out, scores

    def index_bytes(self) -> int:
        """Resident index footprint (vector storage + coarse quantizer + layout sidecars)."""
        return int(self.base_sorted.nbytes + self.cent.nbytes + self.offsets.nbytes
                   + self.order.nbytes + self.row_ids.nbytes)

    def calibrate_nprobe(self, target: float = 0.98, n_samples: int = 256, noise: float = 0.08,
                         probes=(1, 2, 4, 8, 12, 16, 24, 32), k: int = 10, seed: int = 0,
                         val_queries: np.ndarray | None = None):
        """Governed auto-tuning: pick the SMALLEST nprobe that hits a recall target, then stop over-scanning.

        Calibration queries default to samples drawn from the index's own cluster centroids + small noise —
        representative of the data with NO built-in self-match (sampling base points instead gives every query
        a guaranteed near neighbor, which over-estimates recall, especially for the fp16/quantized scans).
        Callers with real traffic should pass val_queries. Ground truth is exact (FAISS-flat IP over the
        resident vectors). Sets self.nprobe and returns (nprobe, recall). No recall is traded below the target;
        if no probe clears it, nprobe falls back to the largest. The 'governed auto-tuning' the charter promises.
        """
        import faiss

        rng = np.random.default_rng(seed)
        if val_queries is not None:
            q = np.ascontiguousarray(val_queries, dtype=np.float32)
        else:
            a = rng.integers(0, self.cent.shape[0], size=n_samples)
            q = self.cent[a] + noise * rng.standard_normal((n_samples, self.dim)).astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
            q = np.ascontiguousarray(q, dtype=np.float32)
        flat = faiss.IndexFlatIP(self.dim); flat.add(self.base_sorted)
        _, tpos = flat.search(q, k)
        truth = self.row_ids[self.order[tpos]]  # exact top-k in row-id space
        chosen, chosen_r = probes[-1], 0.0
        for npb in probes:
            self.nprobe = npb
            pred, _ = self.search(q, k)
            r = float(np.mean([len(set(pred[i].tolist()) & set(truth[i].tolist())) / k for i in range(len(q))]))
            chosen, chosen_r = npb, r
            if r >= target:
                break
        self.nprobe = chosen
        return chosen, chosen_r


class MojoIVFSQ8Backend(MojoIVFBackend):
    """IVF with an int8 scalar-quantized cluster scan + exact float32 rerank.

    Build is identical to MojoIVFBackend plus one streaming pass that quantizes the sorted base to int8
    (global 1/127 scale, via ame_quantize_i8). FIND scans the int8 base (4x denser memory traffic) to a
    candidate set of size cand_cap, then re-scores those candidates in exact float32 -> true top-k. Recall
    is tuned purely via cand_cap (decoupled from nprobe). Inherits calibrate_nprobe (governed auto-tuning).
    """

    def __init__(self, dim: int, nlist: int = 1024, nprobe: int = 32, kmeans: str = "faiss",
                 cand_cap: int | None = None) -> None:
        super().__init__(dim, nlist, nprobe, kmeans)
        self.cand_cap = cand_cap

    def build(self, vectors: np.ndarray, row_ids: np.ndarray | None = None) -> None:
        super().build(vectors, row_ids)
        n = self.base_sorted.shape[0]
        self.base_q8 = np.empty(n * self.dim, dtype=np.int8)
        self.lib.ame_quantize_i8(self._bsp, n, self.dim, self.base_q8.ctypes.data_as(ctypes.c_void_p))
        self._bq8 = self.base_q8.ctypes.data_as(ctypes.c_void_p)

    def search(self, queries: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        cap = self.cand_cap or max(64, 4 * k)
        out = np.empty((q.shape[0], k), dtype=np.int64)
        scores = np.empty((q.shape[0], k), dtype=np.float32)
        oi = np.empty(k, dtype=np.int32)
        od = np.empty(k, dtype=np.float32)
        qq8 = np.empty(self.dim, dtype=np.int8)
        sid = np.empty(cap, dtype=np.int32)
        ssc = np.empty(cap, dtype=np.int32)
        oip, odp = oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p)
        qq8p = qq8.ctypes.data_as(ctypes.c_void_p)
        sidp, sscp = sid.ctypes.data_as(ctypes.c_void_p), ssc.ctypes.data_as(ctypes.c_void_p)
        for i in range(q.shape[0]):
            self.lib.ame_ivf_search_q8(
                self._cp, self.nlist, self._bsp, self._bq8, self._op, self.dim,
                q[i].ctypes.data_as(ctypes.c_void_p), self.nprobe, k, cap,
                qq8p, sidp, sscp, oip, odp,
            )
            out[i] = self.row_ids[self.order[oi.astype(np.int64)]]
            scores[i] = od
        return out, scores


class MojoIVFF16Backend(MojoIVFBackend):
    """IVF with an fp16 cluster scan (centroid selection stays exact f32).

    Build = MojoIVFBackend + one streaming pass converting the sorted base to fp16 (ame_quantize_f16).
    FIND scans the fp16 base (half the bytes streamed, double the SIMD lanes; M-series converts fp16->f32
    in the FMA path) with f32 accumulation. fp16's ~10-bit mantissa preserves unit-vector cosine ranking,
    so recall holds with no residual encoding. Inherits calibrate_nprobe (governed auto-tuning).
    """

    def __init__(self, dim: int, nlist: int = 1024, nprobe: int = 32, kmeans: str = "faiss",
                 cand_cap: int | None = None, auto_target: float | None = None) -> None:
        super().__init__(dim, nlist, nprobe, kmeans)
        self.cand_cap = cand_cap
        self.auto_target = auto_target  # if set, calibrate_nprobe to this recall floor at the end of build

    def build(self, vectors: np.ndarray, row_ids: np.ndarray | None = None) -> None:
        super().build(vectors, row_ids)
        n = self.base_sorted.shape[0]
        self.base_f16 = np.empty(n * self.dim, dtype=np.float16)
        self.lib.ame_quantize_f16(self._bsp, n, self.dim, self.base_f16.ctypes.data_as(ctypes.c_void_p))
        self._bf16 = self.base_f16.ctypes.data_as(ctypes.c_void_p)
        if self.auto_target is not None:
            self.calibrate_nprobe(self.auto_target)

    def index_bytes(self) -> int:
        return super().index_bytes() + self.base_f16.nbytes  # +50%: keeps f32 (rerank) AND f16 (scan)

    def search(self, queries: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        cap = self.cand_cap or max(24, 2 * k + 4)  # sweet spot: enough to catch fp16 mis-ranks, cheap maintenance
        out = np.empty((q.shape[0], k), dtype=np.int64)
        scores = np.empty((q.shape[0], k), dtype=np.float32)
        oi = np.empty(k, dtype=np.int32)
        od = np.empty(k, dtype=np.float32)
        sid = np.empty(cap, dtype=np.int32)
        ssc = np.empty(cap, dtype=np.float32)
        oip, odp = oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p)
        sidp, sscp = sid.ctypes.data_as(ctypes.c_void_p), ssc.ctypes.data_as(ctypes.c_void_p)
        for i in range(q.shape[0]):
            self.lib.ame_ivf_search_f16(
                self._cp, self.nlist, self._bsp, self._bf16, self._op, self.dim,
                q[i].ctypes.data_as(ctypes.c_void_p), self.nprobe, k, cap, sidp, sscp, oip, odp,
            )
            out[i] = self.row_ids[self.order[oi.astype(np.int64)]]
            scores[i] = od
        return out, scores


class MojoNSWBackend:
    """In-process approximate single-layer NSW (HNSW layer-0) — graph BUILT and SEARCHED by real Mojo.

    Build (Mojo `ame_nsw_build`): incremental insertion, ef-beam search to find M neighbours, bidirectional
    links with degree pruning. Search (Mojo `ame_nsw_search`): ef-beam graph traversal -> top-k.
    """

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.lib = _lib()
        self.entry = 0
        self._stamp = 0

    def build(self, vectors: np.ndarray, row_ids: np.ndarray | None = None) -> None:
        self.base = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n = self.base.shape[0]
        self.row_ids = (
            np.asarray(row_ids, dtype=np.int64) if row_ids is not None else np.arange(self.n, dtype=np.int64)
        )
        self.neighbors = np.full(self.n * self.M, -1, dtype=np.int32)
        self.degree = np.zeros(self.n, dtype=np.int32)
        visited = np.zeros(self.n, dtype=np.int32)
        ef = max(self.ef_construction, self.ef_search)
        rid = np.empty(ef, dtype=np.int32)
        rip = np.empty(ef, dtype=np.float32)
        self._bp = self.base.ctypes.data_as(ctypes.c_void_p)
        self._np = self.neighbors.ctypes.data_as(ctypes.c_void_p)
        self._dp = self.degree.ctypes.data_as(ctypes.c_void_p)
        self.lib.ame_nsw_build(
            self._bp, self.n, self.dim, self.M, self.ef_construction,
            self._np, self._dp, visited.ctypes.data_as(ctypes.c_void_p),
            rid.ctypes.data_as(ctypes.c_void_p), rip.ctypes.data_as(ctypes.c_void_p),
        )
        # fresh visited for the search phase (its own monotonic stamp space)
        self._visited = np.zeros(self.n, dtype=np.int32)
        self._vp = self._visited.ctypes.data_as(ctypes.c_void_p)

    def search(self, queries: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        out = np.empty((q.shape[0], k), dtype=np.int64)
        scores = np.empty((q.shape[0], k), dtype=np.float32)
        rid = np.empty(self.ef_search, dtype=np.int32)
        rip = np.empty(self.ef_search, dtype=np.float32)
        oi = np.empty(k, dtype=np.int32)
        od = np.empty(k, dtype=np.float32)
        ridp, ripp = rid.ctypes.data_as(ctypes.c_void_p), rip.ctypes.data_as(ctypes.c_void_p)
        oip, odp = oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p)
        for i in range(q.shape[0]):
            self._stamp += 1
            self.lib.ame_nsw_search(
                self._bp, self.n, self.dim, self.M, self._np, self._dp, self.entry,
                q[i].ctypes.data_as(ctypes.c_void_p), self.ef_search, k, self._vp, self._stamp,
                ridp, ripp, oip, odp,
            )
            out[i] = self.row_ids[oi.astype(np.int64)]
            scores[i] = od
        return out, scores
