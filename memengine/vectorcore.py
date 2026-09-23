"""AME VectorCore (thin MVP) — swappable FAISS backend: Flat / HNSW / IVF-Flat / IVF-PQ.

Mirrors internal/ARCHITECTURE.md (VectorCore) and internal/DECISIONS.md (FAISS-primary, Mojo-native-progressive
+ ANN staging FLAT -> HNSW -> IVF/IVF-PQ). Indexes serialize to disk (faiss.write_index) so the harness
can measure cold-open (read_index from disk) vs warm (resident) — the internal/BENCHMARKS.md cold/warm axis.
"""
from __future__ import annotations

import faiss
import numpy as np

from memengine.partitioned import Eq  # noqa: F401  (re-export: `from memengine.vectorcore import VectorCore, Eq`)

_METRIC = {"ip": faiss.METRIC_INNER_PRODUCT, "l2": faiss.METRIC_L2}


class FaissBackend:
    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        metric: str = "ip",
        *,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        nlist: int = 1024,
        nprobe: int = 16,
        pq_m: int = 16,
        pq_bits: int = 8,
        threads: int = 1,
    ) -> None:
        faiss.omp_set_num_threads(threads)
        self.dim = dim
        self.index_type = index_type
        self.ef_search = ef_search
        self.nprobe = nprobe
        self.row_ids: np.ndarray | None = None
        m = _METRIC[metric]

        if index_type == "flat":
            self.index = faiss.IndexFlat(dim, m)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, M, m)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        elif index_type == "ivf":
            quant = faiss.IndexFlat(dim, m)
            self.index = faiss.IndexIVFFlat(quant, dim, nlist, m)
            self.index.nprobe = nprobe
        elif index_type == "ivfpq":
            quant = faiss.IndexFlat(dim, m)
            self.index = faiss.IndexIVFPQ(quant, dim, nlist, pq_m, pq_bits, m)
            self.index.nprobe = nprobe
        else:
            raise ValueError(f"unknown index_type {index_type!r}")

    def build(self, vectors: np.ndarray, row_ids: np.ndarray | None = None) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if not self.index.is_trained:
            self.index.train(v)
        self.index.add(v)
        self.row_ids = (
            np.asarray(row_ids, dtype=np.int64) if row_ids is not None
            else np.arange(v.shape[0], dtype=np.int64)
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, pos = self.index.search(q, k)
        return self.row_ids[pos], scores

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, row_ids: np.ndarray, *, ef_search: int = 64, nprobe: int = 16) -> "FaissBackend":
        self = cls.__new__(cls)  # bypass __init__: load a serialized index from disk (cold-open)
        self.index = faiss.read_index(path)
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        self.ef_search = ef_search
        self.nprobe = nprobe
        self.index_type = "loaded"
        self.dim = self.index.d
        try:
            self.index.hnsw.efSearch = ef_search
        except AttributeError:
            pass
        try:
            self.index.nprobe = nprobe
        except AttributeError:
            pass
        return self


# Backwards-compatible alias used by run_mvp_bench.py
FaissFlatBackend = FaissBackend


class VectorCore:
    """Swappable vector backend facade (FAISS today, Mojo-native later)."""

    def __init__(self, backend: FaissBackend) -> None:
        self.backend = backend

    @classmethod
    def faiss_flat(cls, dim: int, metric: str = "ip", threads: int = 1) -> "VectorCore":
        return cls(FaissBackend(dim, "flat", metric, threads=threads))

    @classmethod
    def faiss(cls, dim: int, index_type: str = "flat", metric: str = "ip", **kw) -> "VectorCore":
        return cls(FaissBackend(dim, index_type, metric, **kw))

    @classmethod
    def mojo_flat(cls, dim: int) -> "VectorCore":
        """T0 in-process REAL Mojo SIMD kernel via C-ABI (exact FLAT top-k). See mojo_backend.py."""
        from memengine.mojo_backend import MojoFlatBackend
        return cls(MojoFlatBackend(dim))

    @classmethod
    def mojo_ivf(cls, dim: int, nlist: int = 1024, nprobe: int = 32, kmeans: str = "faiss") -> "VectorCore":
        """T0 in-process REAL Mojo IVF kernel via C-ABI (approximate top-k). See mojo_backend.py.

        kmeans="faiss" (BLAS, fast default) or "mojo" (FAISS-free pure-Mojo coarse quantizer, slower build).
        """
        from memengine.mojo_backend import MojoIVFBackend
        return cls(MojoIVFBackend(dim, nlist, nprobe, kmeans))

    @classmethod
    def mojo_ivf_f16(cls, dim: int, nlist: int = 1024, nprobe: int = 32, kmeans: str = "faiss") -> "VectorCore":
        """T0 in-process Mojo IVF with an fp16 cluster scan (half the memory bandwidth; M-series native)."""
        from memengine.mojo_backend import MojoIVFF16Backend
        return cls(MojoIVFF16Backend(dim, nlist, nprobe, kmeans))

    @classmethod
    def telys(cls, dim: int, nlist: int = 1024, nprobe: int = 32, target_recall: float = 0.98,
               kmeans: str = "faiss", cand_cap: int | None = None) -> "VectorCore":
        """Telys — AME's flagship in-process vector index.

        Mojo fp16 cluster scan + exact f32 rerank + governed auto-nprobe (calibrated to `target_recall` once
        at build), over a multi-accumulator SIMD dot. Beats raw FAISS IVF in-process at iso-recall in the
        memory-bound regime. One call builds a fully-tuned index: VectorCore.telys(d, nlist).build(vecs, ids).
        """
        from memengine.mojo_backend import MojoIVFF16Backend
        return cls(MojoIVFF16Backend(dim, nlist, nprobe, kmeans, cand_cap, auto_target=target_recall))

    @classmethod
    def mojo_ivf_sq8(cls, dim: int, nlist: int = 1024, nprobe: int = 32, kmeans: str = "faiss",
                     cand_cap: int | None = None) -> "VectorCore":
        """T0 in-process Mojo IVF: int8 SQ scan + exact f32 rerank.

        MEASURED NON-WIN on tightly-clustered unit vectors — the 1/127 resolution can't separate
        intra-cluster near-ties, so recall needs a large cand_cap whose O(cand_cap) maintenance makes it
        slower than float IVF. Needs residual/IVF-SQ encoding to win; excluded from the optimized default.
        """
        from memengine.mojo_backend import MojoIVFSQ8Backend
        return cls(MojoIVFSQ8Backend(dim, nlist, nprobe, kmeans, cand_cap))

    @classmethod
    def mojo_nsw(cls, dim: int, M: int = 32, ef_construction: int = 200, ef_search: int = 64) -> "VectorCore":
        """T0 in-process REAL Mojo single-layer NSW graph (built + searched in Mojo). See mojo_backend.py."""
        from memengine.mojo_backend import MojoNSWBackend
        return cls(MojoNSWBackend(dim, M, ef_construction, ef_search))

    @classmethod
    def load_faiss(cls, path: str, row_ids: np.ndarray, **kw) -> "VectorCore":
        return cls(FaissBackend.load(path, row_ids, **kw))

    @staticmethod
    def partitioned(dim: int, dtype: str = "f32", cand_cap: int | None = None):
        """PartitionedVectorIndex — filtered vector search as a contiguous-slice scan (see partitioned.py).

        Returns the index directly (its query API takes a partition key / row-set, unlike the kNN backends):
        idx = VectorCore.partitioned(d).build(vecs, ids, keys); idx.search_partition(q, k, key).
        """
        from memengine.partitioned import PartitionedVectorIndex
        return PartitionedVectorIndex(dim, dtype, cand_cap)

    @staticmethod
    def open_partitioned(path: str, verify: bool = True):
        """Reopen a sealed PartitionedVectorIndex from disk (mmap, no rebuild). See partitioned.py."""
        from memengine.partitioned import PartitionedVectorIndex
        return PartitionedVectorIndex.open(path, verify=verify)

    def build(self, vectors: np.ndarray, row_ids: np.ndarray | None = None) -> "VectorCore":
        self.backend.build(vectors, row_ids)
        return self

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.backend.search(queries, k)

    def save(self, path: str) -> None:
        self.backend.save(path)
