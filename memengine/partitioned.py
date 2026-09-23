"""PartitionedVectorIndex — filtered vector search as a contiguous-slice scan (AME's hybrid wedge).

The thesis, narrowly: AME owns the storage layout, so it can make the hot filter key part of the physical
vector layout. A filtered query "nearest WHERE key = x" then becomes an O(1) directory lookup + a sequential
scan of one contiguous block — not thousands of cache-miss random reads (scatter-gather) across the table.

Scope (deliberately tight): exact, single-node, seal-time partitioned. One primary partition key.
  base segment : vectors grouped by primary key; physical->logical row-id remap; partition directory
                 key -> (start, len). Optional fp16/int8 shadow scanned with an exact f32 rerank.
  delta        : append-only new rows + an in-memory per-key mini-partition map; queried after the sealed slice.
  query key=x  : directory lookup -> contiguous slice scan (Mojo) -> merge delta mini-partition -> top-k.
  query attr=y : (non-partition filter) scatter-gather fallback over a caller-supplied row-set, reported honestly.

NOT in scope yet: HNSW/IVF-PQ/DiskANN, distributed, multi-column partitioning, secondary indexes. This wins
because it is simple and physical. Distance dtype is f32 (exact), fp16, or int8 (quantized scan + f32 rerank).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager

import numpy as np

from telys.filters import Eq          # query filter type is a public contract (telys SDK)
from memengine.lexical import LEX_TOKENS_COLUMN as _LEX_COL   # reserved token column (excluded from filter routing)
from memengine.mojo_backend import _lib


class _RWLock:
    """Writer-preferring readers-writer lock: parallel readers, exclusive writer. A WAITING writer blocks new
    readers so in-flight readers drain and the writer proceeds (no writer starvation under continuous reads)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read(self):
        with self._cond:
            while self._writer or self._writers_waiting:   # yield to waiting writers (no starvation)
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _py(v):
    return v.item() if hasattr(v, "item") else v


class PartitionedVectorIndex:
    def __init__(self, dim: int, dtype: str = "f32", cand_cap: int | None = None) -> None:
        assert dtype in ("f32", "fp16", "int8")
        self.dim = dim
        self.dtype = dtype
        self.cand_cap = cand_cap
        self.lib = _lib()
        self._sealed = False
        # Empty-state defaults so a NEVER-BUILT collection (e.g. a repo with 0 extractable symbols) is a valid,
        # compactable, saveable, searchable EMPTY index instead of raising AttributeError. base is an empty array
        # (NOT None — None means the compact-primary i8 serve mode), so scan/save/stats treat it as 0 rows.
        self.n = 0
        self.columns: dict = {}
        self.key_name = "key"
        self.exact_crossover = 1 << 62
        self.build_ns = 0
        self.pdir: dict = {}
        self.base = np.empty((0, dim), np.float32)
        self._bp = self.base.ctypes.data_as(ctypes.c_void_p)
        self.phys_rowid = np.empty(0, np.int64)
        self._log2phys = np.empty(0, np.int64)
        self._base_key = np.empty(0, dtype=object)
        self.compact_primary = False   # read-only i8-only serve artifact (no f32 base on disk); set on open()
        self._delta_vecs: np.ndarray | None = None
        self._delta_rowids: np.ndarray | None = None
        self._delta_keys: np.ndarray | None = None
        self._delta_lsn: np.ndarray | None = None  # per-delta-row write LSN (MVCC versioning)
        self._delta_map: dict = {}
        # MVCC-by-append: monotonic LSN, tombstones (id->delete LSN), latest delta version per id, per-key dirty count
        self._lsn = 0
        self._tomb: dict = {}
        self._delta_latest: dict = {}   # logical id -> (lsn, key) of its newest delta version (current view)
        self._dirty: dict = {}          # partition key -> count of base rows invalidated (overfetch margin)
        self.pivf: dict = {}          # per-partition IVF metadata (key -> {...}) for oversized partitions
        self.plex: dict = {}          # per-partition lexical/BM25 index (key -> LexicalCodeIndex) for mode="lexical"
        self._rw = _RWLock()          # parallel readers, exclusive writer (mutations + compaction)
        self._tls = threading.local() # per-thread reusable top-k scratch (avoids per-query alloc in the hot path)
        self.embedder = None          # optional EmbeddingProvider (AME is usable without one)
        self.profile = None           # pinned vector-space identity; rejects incompatible adds/queries

    def _require_mutable(self) -> None:
        if self.compact_primary:
            raise RuntimeError("this index was opened as a compact-primary (i8-only) serve artifact and is "
                               "read-only — rebuild from source / reopen the full-precision index to mutate")

    # ── build / seal ────────────────────────────────────────────────────────────────────────────
    def _seal(self, vectors: np.ndarray, ids: np.ndarray, keys: np.ndarray) -> None:
        """Sort rows by partition key into the contiguous physical layout + directory + remap + shadow.
        Shared by build() and compact() (the compaction generation swap)."""
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        n = v.shape[0]
        keys = np.asarray(keys)
        order = np.argsort(keys, kind="stable")            # group rows by partition key
        self.base = np.ascontiguousarray(v[order])         # physical layout (f32 source of truth)
        self.phys_rowid = np.asarray(ids, dtype=np.int64)[order]  # physical offset -> logical row id
        log = self.phys_rowid
        self._log2phys = np.empty(int(log.max()) + 1, dtype=np.int64) if n else np.empty(0, np.int64)
        self._log2phys[log] = np.arange(n, dtype=np.int64)  # logical id -> physical offset (dense ids)
        sk = keys[order]
        self._base_key = sk                                # per-physical-row partition key (for MVCC resolution)
        uniq, starts = np.unique(sk, return_index=True)
        self.pdir = {}                                      # partition directory: key -> (start, len)
        for i, key in enumerate(uniq):
            s = int(starts[i]); e = int(starts[i + 1]) if i + 1 < len(uniq) else n
            self.pdir[key.item() if hasattr(key, "item") else key] = (s, e - s)
        self._bp = self.base.ctypes.data_as(ctypes.c_void_p)
        if self.dtype == "fp16":
            self.base_f16 = np.empty(n * self.dim, dtype=np.float16)
            self.lib.ame_quantize_f16(self._bp, n, self.dim, self.base_f16.ctypes.data_as(ctypes.c_void_p))
            self._bq = self.base_f16.ctypes.data_as(ctypes.c_void_p)
        elif self.dtype == "int8":
            self.base_q8 = np.empty(n * self.dim, dtype=np.int8)
            self.lib.ame_quantize_i8(self._bp, n, self.dim, self.base_q8.ctypes.data_as(ctypes.c_void_p))
            self._bq = self.base_q8.ctypes.data_as(ctypes.c_void_p)
        self.n = n
        self._sealed = True

    def _tokens_column(self, texts, columns: dict | None) -> dict:
        """Merge the reserved per-row token-hash column (for the lexical index) into a columns dict. Tokenized
        engine-side (the SDK facade stays engine-free); rides the columns plumbing for MVCC + persistence."""
        from memengine.lexical import LEX_TOKENS_COLUMN, tokenize_hashes
        col = np.empty(len(texts), dtype=object)
        for i, t in enumerate(texts):
            col[i] = np.asarray(tokenize_hashes(str(t)), np.int64)
        out = dict(columns or {})
        out[LEX_TOKENS_COLUMN] = col
        return out

    def build(self, vectors: np.ndarray, ids: np.ndarray, keys: np.ndarray,
              key_name: str = "key", columns: dict | None = None, texts=None) -> "PartitionedVectorIndex":
        self.key_name = key_name
        self.columns = {}
        if texts is not None:
            columns = self._tokens_column(texts, columns)       # retain per-row tokens for build_lexical()
        t0 = time.perf_counter_ns()
        self._seal(vectors, ids, keys)
        self._set_columns(np.asarray(ids, np.int64), columns)   # off-key filter columns, indexed by LOGICAL id
        self.exact_crossover = self._calibrate_crossover() if self.dtype in ("fp16", "int8") else (1 << 62)
        self.build_ns = time.perf_counter_ns() - t0
        return self

    def _set_columns(self, ids: np.ndarray, columns: dict | None) -> None:
        """Store off-key filter columns INDEXED BY LOGICAL ID (object dtype, grows with delta inserts).
        Lets newly-inserted delta rows participate in off-key filters — closing the delta-metadata gap."""
        if not columns:
            return
        ids = np.asarray(ids, np.int64)
        mx = int(ids.max()) if len(ids) else -1
        for col, vals in columns.items():
            arr = self.columns.get(col)
            if arr is None:
                arr = np.full(mx + 1, None, dtype=object)
            elif len(arr) <= mx:
                arr = np.concatenate([arr, np.full(mx + 1 - len(arr), None, dtype=object)])
            arr[ids] = np.asarray(vals, dtype=object)
            self.columns[col] = arr

    def _calibrate_crossover(self, sizes=(1024, 4096, 16384, 65536), trials: int = 25) -> int:
        """Measure the partition size where the quantized scan overtakes exact f32 (per dim/dtype/hardware).

        Times the f32 and shadow plans on real contiguous slices of the base. Returns the smallest size where
        the shadow is faster; if it never wins in range, returns a large value (always prefer exact f32).
        NOT hardcoded — measured at seal time on this machine. Only consulted in approximate mode.
        """
        import time as _t
        rng = np.random.default_rng(0)
        q = np.ascontiguousarray(self.base[rng.integers(0, self.n)], np.float32)
        cross = 1 << 62
        for sz in sizes:
            if sz > self.n:
                break
            def med(dt):
                self._slice_topk(0, sz, q, 10, dt)
                ts = []
                for _ in range(trials):
                    t0 = _t.perf_counter_ns(); self._slice_topk(0, sz, q, 10, dt); ts.append(_t.perf_counter_ns() - t0)
                return float(np.percentile(ts, 50))
            if med(self.dtype) < med("f32"):
                cross = sz
                break
        return cross

    # ── per-partition contiguous slice scan (the win) ─────────────────────────────────────────────
    def _slice_topk(self, start: int, length: int, query: np.ndarray, k: int, plan_dtype: str | None = None):
        """Top-k over base[start:start+length] via the Mojo kernel. Returns (phys_offsets, scores).

        plan_dtype selects the scan precision per query (the planner uses this): "f32" is always available
        (exact); "fp16"/"int8" require the matching shadow built at seal time. Defaults to the index dtype.
        """
        dt = plan_dtype or self.dtype
        if dt != "f32" and not hasattr(self, "_bq"):
            dt = "f32"  # no shadow built -> fall back to exact f32
        qp = self._qptr(query)
        if self.base is None and self.compact_primary == "pq":   # PQ ADC scan -> i8 rerank (the 32x tier)
            return self._slice_topk_pq(start, length, qp, k)
        oi, od, oip, odp = self._topk_scratch(k)
        if self.base is None:    # compact-primary i8 serve: rank on the int8 slab directly, no f32 rerank
            qq8 = np.empty(self.dim, np.int8)
            bq = ctypes.c_void_p(self._bq.value + start * self.dim)
            self.lib.ame_flat_topk_i8(bq, length, self.dim, qp, qq8.ctypes.data_as(ctypes.c_void_p), k, oip, odp)
        elif dt == "f32":
            bp = ctypes.c_void_p(self._bp.value + start * self.dim * 4)
            self.lib.ame_flat_topk(bp, length, self.dim, qp, k, oip, odp)
        elif dt == "fp16":
            cap = min(self.cand_cap or max(24, 2 * k + 4), length)
            sid = np.empty(cap, np.int32); ssc = np.empty(cap, np.float32)
            bp = ctypes.c_void_p(self._bp.value + start * self.dim * 4)
            bq = ctypes.c_void_p(self._bq.value + start * self.dim * 2)
            self.lib.ame_flat_topk_f16(bp, bq, length, self.dim, qp, k, cap,
                                       sid.ctypes.data_as(ctypes.c_void_p), ssc.ctypes.data_as(ctypes.c_void_p), oip, odp)
        else:  # int8
            cap = min(self.cand_cap or max(24, 2 * k + 4), length)
            sid = np.empty(cap, np.int32); ssc = np.empty(cap, np.int32); qq8 = np.empty(self.dim, np.int8)
            bp = ctypes.c_void_p(self._bp.value + start * self.dim * 4)
            bq = ctypes.c_void_p(self._bq.value + start * self.dim * 1)
            self.lib.ame_flat_topk_q8(bp, bq, length, self.dim, qp, qq8.ctypes.data_as(ctypes.c_void_p), k, cap,
                                      sid.ctypes.data_as(ctypes.c_void_p), ssc.ctypes.data_as(ctypes.c_void_p), oip, odp)
        valid = oi >= 0
        return (oi[valid].astype(np.int64) + start), od[valid]

    def _slice_topk_pq(self, start: int, length: int, qp, k: int):
        """compact-primary PQ serve: ADC-scan the codes slice -> top-pool, then i8-rerank the pool -> top-k.
        Returns (phys_offsets, scores). PQ alone is the 32x recall tier; the i8 rerank restores precision
        (and a downstream WordLlama rerank sharpens further). Falls back to PQ top-k if no i8 tier present."""
        m, ks, ds = self._pq_m, self._pq_ksub, self._pq_dsub
        pool = int(min(max(self.cand_cap or 0, 8 * k), length))
        if pool <= 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        lut = np.zeros(m * ks, np.float32)                       # per-call scratch (thread-safe)
        pid = np.full(pool, -1, np.int32); pds = np.zeros(pool, np.float32)
        codes_off = ctypes.c_void_p(self._pq_codes.ctypes.data + start * m)
        self.lib.ame_flat_topk_pq(codes_off, length, m, ks, ds,
                                  self._pq_codebook.ctypes.data_as(ctypes.c_void_p), qp,
                                  lut.ctypes.data_as(ctypes.c_void_p), pool,
                                  pid.ctypes.data_as(ctypes.c_void_p), pds.ctypes.data_as(ctypes.c_void_p))
        sel = pid[pid >= 0]
        if len(sel) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        if getattr(self, "_base_q8_2d", None) is not None:       # i8 rerank of the PQ pool (precision)
            gi8 = np.ascontiguousarray(self._base_q8_2d[start + sel])
            kk = min(k, len(gi8))
            ri = np.full(kk, -1, np.int32); rd = np.zeros(kk, np.float32); qq8 = np.empty(self.dim, np.int8)
            self.lib.ame_flat_topk_i8(gi8.ctypes.data_as(ctypes.c_void_p), len(gi8), self.dim, qp,
                                      qq8.ctypes.data_as(ctypes.c_void_p), kk,
                                      ri.ctypes.data_as(ctypes.c_void_p), rd.ctypes.data_as(ctypes.c_void_p))
            v = ri >= 0
            return (start + sel[ri[v].astype(np.int64)]).astype(np.int64), rd[v]
        order = np.argsort(-pds[:len(sel)])[:k]                  # no i8 tier: PQ top-k directly
        return (start + sel[order]).astype(np.int64), pds[order]

    def _scatter_topk(self, phys: np.ndarray, query: np.ndarray, k: int):
        """Exact f32 top-k over an arbitrary set of physical rows (gather to contiguous, then scan)."""
        if self.base is None:
            raise RuntimeError("off-key/scatter filters are not supported on a compact-primary (i8-only) "
                               "index — query by the partition key, or reopen the full-precision index")
        if len(phys) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        g = np.ascontiguousarray(self.base[phys])          # GATHER (random reads) — the fallback cost
        kk = min(k, len(g))
        oi = np.empty(kk, np.int32); od = np.empty(kk, np.float32)
        self.lib.ame_flat_topk(g.ctypes.data_as(ctypes.c_void_p), len(g), self.dim,
                               np.ascontiguousarray(query, np.float32).ctypes.data_as(ctypes.c_void_p), kk,
                               oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p))
        valid = oi >= 0
        return phys[oi[valid].astype(np.int64)], od[valid]

    def _delta_topk(self, key, query: np.ndarray, k: int):
        idx = self._delta_map.get(key.item() if hasattr(key, "item") else key)
        if not idx:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        idx = np.asarray(idx, np.int64)
        g = np.ascontiguousarray(self._delta_vecs[idx])
        kk = min(k, len(g))
        oi = np.empty(kk, np.int32); od = np.empty(kk, np.float32)
        self.lib.ame_flat_topk(g.ctypes.data_as(ctypes.c_void_p), len(g), self.dim,
                               np.ascontiguousarray(query, np.float32).ctypes.data_as(ctypes.c_void_p), kk,
                               oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p))
        valid = oi >= 0
        return self._delta_rowids[idx[oi[valid].astype(np.int64)]], od[valid]

    @staticmethod
    def _merge(ids_a, sc_a, ids_b, sc_b, k):
        ids = np.concatenate([ids_a, ids_b]); sc = np.concatenate([sc_a, sc_b])
        if len(ids) == 0:
            return ids.astype(np.int64), sc
        top = np.argsort(-sc)[:k]
        return ids[top].astype(np.int64), sc[top]

    # ── query ─────────────────────────────────────────────────────────────────────────────────────
    def search_partition(self, query: np.ndarray, k: int, key):
        """WHERE partition_key == key  +  vector. O(1) directory lookup -> contiguous slice scan (+delta)."""
        ent = self.pdir.get(key.item() if hasattr(key, "item") else key)
        if ent is None:
            s_ids, s_sc = np.empty(0, np.int64), np.empty(0, np.float32)
        else:
            phys, s_sc = self._slice_topk(ent[0], ent[1], query, k)
            s_ids = self.phys_rowid[phys]
        d_ids, d_sc = self._delta_topk(key, query, k)
        return self._merge(s_ids, s_sc, d_ids, d_sc, k)

    def _gather_latest(self, lids, S=None):
        """Latest VISIBLE vector per logical id (base or delta), skipping deleted/absent. (vecs, kept ids)."""
        vecs, keep = [], []
        for lid in np.asarray(lids, np.int64).tolist():
            lat_lsn, _ = self._visible_key(lid, S)
            if lat_lsn < 0:
                continue
            dl = self._delta_latest.get(lid)
            if dl is not None and dl[0] == lat_lsn:
                vecs.append(self._delta_vecs[dl[2]])
            else:
                p = self._base_phys(lid)
                if p < 0:
                    continue
                vecs.append(self.base[p])
            keep.append(lid)
        if not keep:
            return np.empty((0, self.dim), np.float32), np.empty(0, np.int64)
        return np.ascontiguousarray(np.stack(vecs), np.float32), np.asarray(keep, np.int64)

    def search_filter(self, query: np.ndarray, k: int, rowset: np.ndarray, as_of=None):
        """WHERE (non-partition attr) -> caller supplies matching logical ids. MVCC-aware scatter over base+delta
        (each row's latest visible version; deleted rows excluded). Exact f32 scan of the gathered survivors."""
        if self.base is None:
            raise RuntimeError("off-key/scatter filters are not supported on a compact-primary (i8-only) "
                               "index — query by the partition key, or reopen the full-precision index")
        g, keep = self._gather_latest(rowset, as_of)
        if len(keep) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        kk = min(k, len(g)); oi = np.empty(kk, np.int32); od = np.empty(kk, np.float32)
        self.lib.ame_flat_topk(g.ctypes.data_as(ctypes.c_void_p), len(g), self.dim, self._qptr(query), kk,
                               oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p))
        v = oi >= 0
        return keep[oi[v].astype(np.int64)], od[v]

    # ── planner + natural query surface (where / explain / batch) ─────────────────────────────────
    PLAN_BY_DTYPE = {"f32": "PartitionSliceExactF32", "fp16": "PartitionSliceF16RerankF32",
                     "int8": "PartitionSliceQ8RerankF32"}

    def choose_plan(self, where, target_recall: float = 1.0) -> dict:
        """PartitionAwareVectorPlanner: pick a physical plan from the filter, partition size, and target.

        Plans: PartitionSlice{ExactF32,F16RerankF32,Q8RerankF32} · ScatterGatherExact · FullScanMask
        (PartitionIVFRerankF32 is reserved for oversized partitions — not built yet, never selected here).
        Crossover to a quantized scan is the MEASURED self.exact_crossover, used only in approximate mode.
        """
        if where is None:
            return {"plan": "FullScanMask", "fallback": False}
        if isinstance(where, Eq) and where.column == self.key_name:
            ent = self.pdir.get(_py(where.value))
            rows = ent[1] if ent else 0
            approx = target_recall < 1.0
            if approx and _py(where.value) in self.pivf:          # oversized partition with an IVF sublayout
                return {"plan": "PartitionIVFRerankF32", "scan_dtype": "ivf", "exact": False,
                        "partition_entry": ent, "partition_value": where.value, "fallback": False}
            if approx and hasattr(self, "_bq") and rows >= self.exact_crossover:
                dt = self.dtype; exact = False          # quantized shadow above the measured crossover
            else:
                dt = "f32"; exact = True
            return {"plan": self.PLAN_BY_DTYPE[dt], "scan_dtype": dt, "exact": exact,
                    "partition_entry": ent, "partition_value": where.value, "fallback": False}
        if isinstance(where, Eq) and where.column in self.columns and where.column != _LEX_COL:
            return {"plan": "ScatterGatherExact", "column": where.column, "value": where.value,
                    "exact": True, "fallback": True,
                    "fallback_reason": f"{where.column} is not the physical partition key"}
        return {"plan": "FullScanMask", "fallback": True, "fallback_reason": "no usable filter column"}

    def search(self, query: np.ndarray, k: int = 10, where=None, target_recall: float = 1.0,
               explain: bool = False, as_of: int | None = None):
        """Natural filtered vector query (read-locked: parallel with other readers, excluded by writers).

        idx.search(q, k=20, where=Eq("tenant_id", "acme"), explain=True)
        as_of=<lsn from snapshot()> reads a consistent historical view (MVCC); None = latest.
        """
        with self._rw.read():
            return self._search(query, k, where, target_recall, explain, as_of)

    def _search(self, query: np.ndarray, k: int = 10, where=None, target_recall: float = 1.0,
                explain: bool = False, as_of: int | None = None):
        if isinstance(where, dict):
            if len(where) != 1:
                raise ValueError("dict `where` supports a single equality; pass Eq(...) (composites are future)")
            (c, v), = where.items(); where = Eq(c, v)
        plan = self.choose_plan(where, target_recall)
        name = plan["plan"]
        ex = None
        if name.startswith("Partition"):            # PartitionSlice{Exact,F16,Q8} or PartitionIVFRerankF32
            ent = plan["partition_entry"]; val = plan["partition_value"]
            ids, sc, base_rows = self._resolve_partition(query, k, val, ent, plan["scan_dtype"], as_of)
            if explain:
                ex = {"plan": name, "partition_key": self.key_name, "partition_value": _py(val),
                      "base_rows": int(base_rows), "delta_rows": len(self._delta_map.get(_py(val), [])),
                      "exact": plan["exact"], "fallback": False,
                      "as_of_lsn": (self._lsn if as_of is None else as_of), "mvcc": self._mutated()}
                if plan["scan_dtype"] == "ivf" and _py(val) in self.pivf:
                    e = self.pivf[_py(val)]; npb, nl = int(e["nprobe"]), int(e["nlist"])
                    # IVF reranks only the vectors in the probed clusters — estimate it from the probe ratio
                    # (NOT a static cand_cap, which would misreport the IVF scan scope).
                    ex["nprobe"] = npb; ex["nlist"] = nl
                    ex["candidates_reranked"] = int(min(base_rows, round(base_rows * npb / max(1, nl))))
                else:
                    cap = k if plan["scan_dtype"] == "f32" else (self.cand_cap or max(24, 2 * k + 4))
                    ex["candidates_reranked"] = int(min(cap, base_rows))
        elif name == "ScatterGatherExact":
            col, val = plan["column"], plan["value"]
            rowset = np.where(self.columns[col] == val)[0]
            ids, sc = self.search_filter(query, k, rowset, as_of)
            if explain:
                ex = {"plan": name, "filter": f"{col} == {val!r}", "candidate_rows": int(len(rowset)),
                      "exact": True, "fallback": True, "fallback_reason": plan["fallback_reason"]}
        else:  # FullScanMask — exact kNN over the whole base (no partition pruning)
            if self._mutated():
                # MVCC-aware full scan: over every live logical id (base+delta, deleted/superseded excluded),
                # not the raw sealed base — otherwise where=None returns stale vectors + resurrected deletes.
                rowset = np.asarray(self.live_ids(None), np.int64)
                ids, sc = self.search_filter(query, k, rowset, as_of)
            else:
                # whole-base exact scan via _slice_topk(0, n) — also picks the compact i8 path when base is None
                phys, sc = self._slice_topk(0, self.n, query, min(k, self.n))
                ids = self.phys_rowid[phys]
            if explain:
                ex = {"plan": name, "base_rows": int(self.n), "exact": True,
                      "fallback": plan.get("fallback", False), "mvcc": self._mutated()}
        return (ids, sc, ex) if explain else (ids, sc)

    def search_batch(self, queries: np.ndarray, k: int = 10, where=None, keys=None,
                     target_recall: float = 1.0, explain: bool = False):
        """Batch query. Pass a single `where`, or per-query `keys` (each routed as Eq(key_name, keys[i]))."""
        out = []
        for i in range(len(queries)):
            w = Eq(self.key_name, keys[i]) if keys is not None else where
            out.append(self.search(queries[i], k, w, target_recall, explain))
        return out

    # ── read-back: provenance with a hit + live-id enumeration (self-describing on read) ───────────
    def columns_for(self, lids) -> list:
        """Per-row off-key column values for the given LOGICAL ids — provenance returned alongside a query,
        so a consumer needn't keep a side map of id -> metadata. (The partition key is in explain.)"""
        from memengine.lexical import LEX_TOKENS_COLUMN
        items = [(n, a) for n, a in self.columns.items() if n != LEX_TOKENS_COLUMN]  # hide the reserved token col
        out = []
        for lid in np.asarray(lids, np.int64).tolist():
            out.append({name: (_py(arr[lid]) if 0 <= lid < len(arr) else None) for name, arr in items})
        return out

    def live_ids(self, where=None) -> list:
        """LOGICAL ids currently visible (not deleted/superseded), optionally filtered by an Eq on the
        partition key (a scope) or an off-key column. Lets a consumer enumerate what a scope contains
        (e.g. to diff against on re-index) without persisting its own id set."""
        with self._rw.read():
            if isinstance(where, dict):
                if len(where) != 1:
                    raise ValueError("dict `where` supports a single equality; pass Eq(...)")
                (c, v), = where.items(); where = Eq(c, v)
            cand: set = set()
            if getattr(self, "phys_rowid", None) is not None:
                cand.update(int(x) for x in self.phys_rowid.tolist())
            cand.update(int(x) for x in self._delta_latest.keys())
            out = []
            for lid in cand:
                lat_lsn, key = self._visible_key(lid, None)
                if lat_lsn < 0:
                    continue                                   # deleted / superseded away
                if where is not None:
                    if where.column == self.key_name:
                        if _py(key) != _py(where.value):
                            continue
                    elif where.column in self.columns and where.column != _LEX_COL:
                        arr = self.columns[where.column]
                        if not (0 <= lid < len(arr)) or _py(arr[lid]) != _py(where.value):
                            continue
                    else:
                        return []                              # unknown/reserved filter column -> no matches
                out.append(lid)
            return out

    # ── text surface (optional embedder; AME is fully usable with raw vectors) ─────────────────────
    def attach_embedder(self, embedder) -> "PartitionedVectorIndex":
        """Attach a provider; enforces the pinned vector-space identity (same dim ≠ same space)."""
        if self.profile is not None and not embedder.profile.compatible_with(self.profile):
            raise ValueError(f"embedder space {embedder.profile.space_id()} != index space {self.profile.space_id()}")
        if embedder.profile.dimension != self.dim:
            raise ValueError(f"embedder dim {embedder.profile.dimension} != index dim {self.dim}")
        self.embedder = embedder
        if self.profile is None:
            self.profile = embedder.profile
        return self

    def build_texts(self, texts, ids, keys, embedder, key_name: str = "key", columns: dict | None = None):
        """Embed documents with the provider, pin its profile, then seal-time partition by key."""
        self.attach_embedder(embedder)
        return self.build(embedder.embed_documents(list(texts)), ids, keys, key_name, columns)

    def add_texts(self, texts, ids, keys) -> None:
        """Embed + append to the delta. Requires an attached embedder (profile already pinned)."""
        if self.embedder is None:
            raise RuntimeError("no embedder attached — use build_texts() or attach_embedder() first")
        self.add(self.embedder.embed_documents(list(texts)), ids, keys)

    def search_text(self, text: str, k: int = 10, where=None, target_recall: float = 1.0, explain: bool = False):
        """Embed the query with the SAME provider (same space) and run the planned filtered search."""
        if self.embedder is None:
            raise RuntimeError("no embedder attached — use attach_embedder() or pass a vector to search()")
        qv = self.embedder.embed_queries([text])[0]
        return self.search(qv, k, where, target_recall, explain)

    # ── mutation (append-only delta with per-key mini-partitions) ──────────────────────────────────
    def _base_phys(self, lid: int) -> int:
        """Physical offset of a logical id in the sealed base, or -1 if not a base row (dense-id assumption)."""
        if 0 <= lid < len(self._log2phys):
            p = int(self._log2phys[lid])
            if 0 <= p < self.n and int(self.phys_rowid[p]) == lid:
                return p
        return -1

    def _append_delta(self, vectors, ids, keys, lsn: int) -> None:
        v = np.ascontiguousarray(vectors, np.float32)
        r = np.asarray(ids, np.int64); keys = np.asarray(keys)
        ln = np.full(len(r), lsn, np.int64)
        if self._delta_vecs is None:
            self._delta_vecs, self._delta_rowids, self._delta_keys, self._delta_lsn = v, r, keys, ln
        else:
            self._delta_vecs = np.vstack([self._delta_vecs, v])
            self._delta_rowids = np.concatenate([self._delta_rowids, r])
            self._delta_keys = np.concatenate([self._delta_keys, keys])
            self._delta_lsn = np.concatenate([self._delta_lsn, ln])
        base = len(self._delta_rowids) - len(r)
        for i, key in enumerate(keys):
            self._delta_map.setdefault(_py(key), []).append(base + i)
            self._delta_latest[int(r[i])] = (lsn, _py(key), base + i)  # newest version: (lsn, key, delta idx)

    def insert(self, vectors, ids, keys, columns=None, texts=None) -> int:
        """Insert new logical rows (existing or new partitions); columns keep delta rows in off-key filters.
        Returns the write LSN."""
        self._require_mutable()
        if texts is not None:
            columns = self._tokens_column(texts, columns)
        with self._rw.write():
            self._lsn += 1
            self._append_delta(vectors, ids, keys, self._lsn)
            self._set_columns(np.asarray(ids, np.int64), columns)
            return self._lsn

    add = insert  # backward-compatible alias

    def current_key(self, lid: int):
        """The partition the logical id lives in right now (latest version), or None if absent/deleted."""
        dl = self._delta_latest.get(int(lid))
        base_lsn = 0 if self._base_phys(lid) >= 0 else -1
        lat_lsn = max(base_lsn, dl[0] if dl else -1)
        if lat_lsn < 0:
            return None
        t = self._tomb.get(int(lid), -1)
        if t >= lat_lsn:
            return None
        return dl[1] if (dl and dl[0] == lat_lsn) else _py(self._base_key[self._base_phys(lid)])

    def delete(self, ids) -> int:
        """Tombstone logical rows (base or delta). Returns the delete LSN."""
        self._require_mutable()
        with self._rw.write():
            self._lsn += 1
            for lid in np.asarray(ids).tolist():
                ck = self.current_key(lid)
                if ck is not None:
                    self._dirty[ck] = self._dirty.get(ck, 0) + 1
                self._tomb[int(lid)] = self._lsn
            return self._lsn

    def update(self, ids, vectors, keys=None, columns=None, texts=None) -> int:
        """Update vectors (and optionally the partition key + off-key columns): tombstone-old is implicit via
        LSN supersession; the new version is appended to the (possibly new) partition's delta. Returns the LSN."""
        self._require_mutable()
        if texts is not None:
            columns = self._tokens_column(texts, columns)
        with self._rw.write():
            self._lsn += 1
            ids = np.asarray(ids, np.int64)
            if keys is None:                                   # vector-only update: keep each id's current key
                keys = np.array([self.current_key(int(i)) for i in ids])
            keys = np.asarray(keys)
            for i, lid in enumerate(ids.tolist()):             # mark the OLD partition dirty (overfetch margin)
                ock = self.current_key(lid)
                if ock is not None and ock != _py(keys[i]):
                    self._dirty[ock] = self._dirty.get(ock, 0) + 1
                if self._tomb.get(lid, -1) >= 0:               # re-insert after delete resurrects (new lsn wins)
                    self._tomb.pop(lid, None)
            self._append_delta(vectors, ids, keys, self._lsn)
            self._set_columns(ids, columns)
            return self._lsn

    def snapshot(self) -> int:
        """A consistent read watermark (current max LSN). Pass to search(..., as_of=snap)."""
        return self._lsn

    def compact(self) -> dict:
        """Fold the delta + tombstones into a fresh sealed generation; GC deleted/superseded rows.

        Live set = base rows neither superseded nor tombstoned + each id's latest live delta version,
        re-partitioned by current key. LSN stays monotonic; delta/tombstones/pivf reset. Runs under the
        write lock so no live reader's mapped slice is invalidated (readers and compaction are exclusive).
        (Lock-free build-then-swap COW is a future optimization; the no-invalidation guarantee holds now.)
        """
        self._require_mutable()
        with self._rw.write():
            before = self.n + (0 if self._delta_vecs is None else len(self._delta_rowids))
            # A never-built collection (0 extractable symbols) has no sealed base — treat base rows as empty so
            # compact() folds only the delta (or seals an empty generation), instead of AttributeError on self.base.
            have_base = self.n > 0 and getattr(self, "base", None) is not None
            if have_base:
                superseded = set(self._delta_latest.keys()) | set(self._tomb.keys())
                if superseded:
                    sup = np.fromiter(superseded, np.int64, len(superseded))
                    base_mask = ~np.isin(self.phys_rowid, sup)
                else:
                    base_mask = np.ones(self.n, bool)
                bidx = np.where(base_mask)[0]
                vv = [self.base[bidx]]; ii = [self.phys_rowid[bidx]]; kk = [np.asarray(self._base_key)[bidx]]
            else:
                vv = [np.empty((0, self.dim), np.float32)]
                ii = [np.empty(0, np.int64)]; kk = [np.empty(0, dtype=object)]
            didx = [info[2] for lid, info in self._delta_latest.items() if self._tomb.get(lid, -1) < info[0]]
            if didx:
                d = np.asarray(didx, np.int64)
                vv.append(self._delta_vecs[d]); ii.append(self._delta_rowids[d]); kk.append(np.asarray(self._delta_keys)[d])
            V = np.vstack(vv) if len(vv) > 1 else vv[0]
            ids = np.concatenate(ii); keys = np.concatenate([np.asarray(x, dtype=object) for x in kk])
            self._seal(V, ids, keys)
            self._delta_vecs = self._delta_rowids = self._delta_keys = self._delta_lsn = None
            self._delta_map = {}; self._delta_latest = {}; self._tomb = {}; self._dirty = {}
            self.pivf = {}                                  # IVF sublayouts invalidated by the re-seal
            self.plex = {}                                  # lexical sublayouts invalidated too (rebuild after)
            return {"rows_before": int(before), "rows_after": int(self.n),
                    "reclaimed": int(before - self.n), "lsn": int(self._lsn)}

    # ── IVF-within-partition (the oversized-partition fix) ─────────────────────────────────────────
    def build_partition_ivf(self, min_rows: int = 20000, nlist: int | None = None,
                            target_recall: float = 0.98, max_nprobe: int = 64) -> dict:
        """Build a per-partition IVF sublayout for partitions with >= min_rows (heterogeneous strategy:
        small partitions stay exact slice scans; oversized ones get IVF). Reorders each big partition's rows
        by sub-cluster in place and calibrates its nprobe to target_recall (measured per partition). Run on a
        RAM index (post-build / pre-save); a reopened mmap base is copied to RAM. Cleared by compact()."""
        self._require_mutable()
        import faiss as _f
        with self._rw.write():
            if isinstance(self.base, np.memmap):
                self.base = np.array(self.base)            # IVF reorder needs a writable base
                self._bp = self.base.ctypes.data_as(ctypes.c_void_p)
            rng = np.random.default_rng(0)
            built = {}
            for key, (start, L) in list(self.pdir.items()):
                if L < min_rows:
                    continue
                nl = nlist or int(min(4096, max(16, round(L ** 0.5))))
                sl = np.ascontiguousarray(self.base[start:start + L])
                km = _f.Kmeans(self.dim, nl, niter=10, seed=42, verbose=False); km.train(sl)
                cents = np.ascontiguousarray(km.centroids.reshape(nl, self.dim), np.float32)
                assign = np.argmax(sl @ cents.T, axis=1)
                sub = np.argsort(assign, kind="stable")    # group the partition's rows by sub-cluster
                self.base[start:start + L] = sl[sub]
                ph = self.phys_rowid[start:start + L].copy(); self.phys_rowid[start:start + L] = ph[sub]
                self._log2phys[self.phys_rowid[start:start + L]] = np.arange(start, start + L, dtype=np.int64)
                counts = np.bincount(assign, minlength=nl)
                offs = np.zeros(nl + 1, np.int32); offs[1:] = np.cumsum(counts).astype(np.int32)
                npb = self._calibrate_partition_nprobe(start, L, cents, offs, nl, target_recall, max_nprobe, rng)
                self.pivf[_py(key)] = {"cents": cents, "offsets": offs, "nlist": nl, "nprobe": npb}
                built[_py(key)] = {"rows": L, "nlist": nl, "nprobe": npb}
            if hasattr(self, "base_f16"):                  # refresh shadow to match the reordered base
                self.lib.ame_quantize_f16(self._bp, self.n, self.dim, self.base_f16.ctypes.data_as(ctypes.c_void_p))
            if hasattr(self, "base_q8"):
                self.lib.ame_quantize_i8(self._bp, self.n, self.dim, self.base_q8.ctypes.data_as(ctypes.c_void_p))
            return built

    def _calibrate_partition_nprobe(self, start, L, cents, offs, nl, target, max_np, rng, samples=64, k=10):
        import faiss as _f
        sl = np.ascontiguousarray(self.base[start:start + L])
        qi = rng.integers(0, L, size=min(samples, L))
        q = sl[qi] + 0.05 * rng.standard_normal((len(qi), self.dim)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12; q = np.ascontiguousarray(q, np.float32)
        flat = _f.IndexFlatIP(self.dim); flat.add(sl); _, truth = flat.search(q, k)  # slice-relative positions
        cp = cents.ctypes.data_as(ctypes.c_void_p); op = offs.ctypes.data_as(ctypes.c_void_p)
        bp = ctypes.c_void_p(self._bp.value + start * self.dim * 4)
        probes = [p for p in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64) if p <= min(nl, max_np)]
        for npb in probes:
            hit = 0
            for i in range(len(q)):
                oi = np.empty(k, np.int32); od = np.empty(k, np.float32)
                self.lib.ame_ivf_search(cp, nl, bp, op, self.dim,
                                        q[i].ctypes.data_as(ctypes.c_void_p), npb, k,
                                        oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p))
                hit += len(set(oi[oi >= 0].tolist()) & set(truth[i].tolist()))
            if hit / (len(q) * k) >= target:
                return npb
        return probes[-1]

    # ── per-partition lexical/BM25 index (mode="lexical") ─────────────────────────────────────────
    def _live_by_key(self, S=None):
        """{partition key -> [live logical ids]} at snapshot S. Lock-free (callers hold the lock). Mirrors
        live_ids' visibility but groups by the CURRENT partition (from _visible_key), for per-partition builds."""
        from collections import defaultdict
        cand: set = set()
        if getattr(self, "phys_rowid", None) is not None:
            cand.update(int(x) for x in self.phys_rowid.tolist())
        cand.update(int(x) for x in self._delta_latest.keys())
        by_key: dict = defaultdict(list)
        for lid in cand:
            lat_lsn, key = self._visible_key(lid, S)
            if lat_lsn >= 0:
                by_key[_py(key)].append(lid)
        return by_key

    def build_lexical(self, k1: float = 1.8, b: float = 1.0) -> dict:
        """Fit a per-partition BM25 lexical index (self.plex) over the LIVE docs' retained token hashes
        (the reserved LEX_TOKENS_COLUMN, populated at add time). Mirrors build_partition_ivf: eager, per
        partition, cleared by compact(). Requires tokens were retained (lexical-enabled ingest)."""
        from memengine.lexical import LEX_TOKENS_COLUMN, LexicalCodeIndex
        with self._rw.write():                              # aux-index build (reads tokens, writes self.plex);
            self.plex = {}                                 # not a base mutation, so allowed on reopened indexes
            toks = self.columns.get(LEX_TOKENS_COLUMN)
            if toks is None:                               # no tokens retained (empty / never lexical-ingested)
                return {"partitions": 0, "docs": 0}
            built = {}
            for key, lids in self._live_by_key(None).items():
                ids, hashes = [], []
                for lid in lids:
                    t = toks[lid] if 0 <= lid < len(toks) else None
                    if t is None:
                        continue
                    ids.append(lid)
                    hashes.append(np.asarray(t, np.int64).tolist())
                if not ids:
                    continue
                self.plex[key] = LexicalCodeIndex(k1=k1, b=b).build_from_hashes(ids, hashes)
                built[key] = len(ids)
            return {"partitions": len(self.plex), "docs": int(sum(built.values()))}

    def _lex_adhoc(self, lids, k1: float, b: float):
        """Fit a throwaway LexicalCodeIndex over an arbitrary live logical-id set (off-key / no-filter paths)."""
        from memengine.lexical import LEX_TOKENS_COLUMN, LexicalCodeIndex
        toks = self.columns.get(LEX_TOKENS_COLUMN)
        if toks is None:
            return None
        ids, hashes = [], []
        for lid in lids:
            t = toks[lid] if 0 <= lid < len(toks) else None
            if t is not None:
                ids.append(int(lid)); hashes.append(np.asarray(t, np.int64).tolist())
        return LexicalCodeIndex(k1=k1, b=b).build_from_hashes(ids, hashes) if ids else None

    def _lex_delta_lids(self, key, as_of):
        """Live logical ids whose CURRENT version is a delta row in partition `key` (inserts + updates-in)."""
        out = []
        for di in self._delta_map.get(_py(key), []):
            lid = int(self._delta_rowids[di])
            lat_lsn, lat_key = self._visible_key(lid, as_of)
            if lat_lsn >= 0 and _py(lat_key) == _py(key):
                out.append(lid)
        return out

    def _lex_partition_hits(self, key, text, k, as_of, k1, b):
        """BM25 hits for partition `key`, MVCC-correct WITHOUT a rebuild: pre-built plex for base rows still
        current (skip any superseded by a delta version), + an ad-hoc index over the partition's live delta
        rows (inserts/updates carry current tokens). Mirrors the vector path's base+delta merge."""
        hits = []
        idx = self.plex.get(_py(key))
        if idx is not None:
            for lid, sc in idx.search(text, k + self._margin(key)):
                if int(lid) not in self._delta_latest:     # base version still current (not updated since build)
                    hits.append((int(lid), float(sc)))
        dl = self._lex_delta_lids(key, as_of)
        adhoc = self._lex_adhoc(dl, k1, b) if dl else None
        if adhoc is not None:
            hits.extend((int(lid), float(sc)) for lid, sc in adhoc.search(text, max(k, 1)))
        return hits

    def search_lexical(self, text: str, k: int = 10, where=None, explain: bool = False, as_of=None,
                       k1: float = 1.8, b: float = 1.0):
        """BM25 code search over a filtered scope. Eq(partition key) -> the pre-built per-partition index + a
        delta merge (the wedge); no filter -> union over all partitions; Eq(off-key column) -> an ad-hoc index
        over the matching LIVE rows. MVCC-correct: base rows superseded by an update are skipped, delta rows
        carry current tokens, and every hit is re-checked through _visible_key so tombstoned/superseded/
        moved-out rows never surface. Returns (ids, scores) (logical ids) or (ids, scores, explain)."""
        with self._rw.read():
            if isinstance(where, dict):
                if len(where) != 1:
                    raise ValueError("dict `where` supports a single equality; pass Eq(...)")
                (c, v), = where.items(); where = Eq(c, v)
            expect_key = None
            if isinstance(where, Eq) and where.column == self.key_name:
                expect_key = _py(where.value)
                hits = self._lex_partition_hits(expect_key, text, k, as_of, k1, b)
                plan = "PartitionLexicalBM25"
            elif where is None:                                # union over every partition (plex + delta merge)
                keys = set(self.plex.keys()) | {_py(kk) for kk in self._delta_map.keys()}
                hits = [h for key in keys for h in self._lex_partition_hits(key, text, k, as_of, k1, b)]
                plan = "FullScanLexicalBM25"
            elif isinstance(where, Eq) and where.column in self.columns and where.column != _LEX_COL:
                arr = self.columns[where.column]               # off-key: ad-hoc over the matching LIVE rows
                live = {lid for lids in self._live_by_key(as_of).values() for lid in lids}
                lids = [lid for lid in np.where(arr == where.value)[0].tolist() if lid in live]
                adhoc = self._lex_adhoc(lids, k1, b)
                hits = adhoc.search(text, max(k, 1)) if adhoc is not None else []
                plan = "ScatterGatherLexicalBM25"
            else:
                raise ValueError(f"lexical search: unusable filter column {getattr(where, 'column', where)!r}")
            out_i, out_s, seen = [], [], set()
            for lid, sc in sorted(hits, key=lambda x: -x[1]):  # MVCC filter: keep latest-visible-in-scope only
                lid = int(lid)
                if lid in seen:
                    continue
                lat_lsn, lat_key = self._visible_key(lid, as_of)
                if lat_lsn < 0 or (expect_key is not None and _py(lat_key) != expect_key):
                    continue
                seen.add(lid); out_i.append(lid); out_s.append(float(sc))
                if len(out_i) >= k:
                    break
            ids = np.asarray(out_i, np.int64); scores = np.asarray(out_s, np.float32)
            if explain:
                ex = {"plan": plan, "mode": "lexical", "partition_key": self.key_name,
                      "partition_value": expect_key, "hits": int(len(ids)),
                      "as_of_lsn": (self._lsn if as_of is None else as_of), "mvcc": self._mutated()}
                return ids, scores, ex
            return ids, scores

    # ── MVCC visibility resolution ────────────────────────────────────────────────────────────────
    def _delta_latest_asof(self, lid: int, S: int):
        """Newest delta version of id with lsn<=S -> (lsn, key), or (-1, None). Used for historical snapshots."""
        if self._delta_rowids is None:
            return -1, None
        m = (self._delta_rowids == lid) & (self._delta_lsn <= S)
        if not m.any():
            return -1, None
        j = int(np.argmax(np.where(m, self._delta_lsn, -1)))
        return int(self._delta_lsn[j]), _py(self._delta_keys[j])

    def _visible_key(self, lid: int, S):
        """Latest VISIBLE (lsn, key) of a logical id at snapshot S (None = current). (-1, None) if deleted/absent."""
        p = self._base_phys(lid)
        base_lsn = 0 if p >= 0 else -1
        if S is None:
            dl = self._delta_latest.get(lid)
            d_lsn, d_key = (dl[0], dl[1]) if dl else (-1, None)
        else:
            d_lsn, d_key = self._delta_latest_asof(lid, S)
        if d_lsn >= 0 and d_lsn >= base_lsn:
            lat_lsn, lat_key = d_lsn, d_key
        else:
            lat_lsn, lat_key = base_lsn, (_py(self._base_key[p]) if p >= 0 else None)
        t = self._tomb.get(lid, -1)
        if t >= 0 and (S is None or t <= S) and t >= lat_lsn:
            return -1, None
        return lat_lsn, lat_key

    def _margin(self, key) -> int:
        return self._dirty.get(_py(key), 0) + len(self._delta_map.get(_py(key), []))

    def _delta_cands(self, key, query, k: int, S):
        """Delta candidates in partition `key` (lsn<=S): (logical ids, scores, lsns)."""
        idx = self._delta_map.get(_py(key))
        if not idx:
            return np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.int64)
        idx = np.asarray(idx, np.int64)
        if S is not None:
            idx = idx[self._delta_lsn[idx] <= S]
        if len(idx) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.int64)
        g = np.ascontiguousarray(self._delta_vecs[idx])
        kk = min(k, len(g)); oi = np.empty(kk, np.int32); od = np.empty(kk, np.float32)
        self.lib.ame_flat_topk(g.ctypes.data_as(ctypes.c_void_p), len(g), self.dim,
                               np.ascontiguousarray(query, np.float32).ctypes.data_as(ctypes.c_void_p), kk,
                               oi.ctypes.data_as(ctypes.c_void_p), od.ctypes.data_as(ctypes.c_void_p))
        v = oi >= 0; sel = idx[oi[v].astype(np.int64)]
        return self._delta_rowids[sel], od[v], self._delta_lsn[sel]

    def _mutated(self) -> bool:
        return bool(self._tomb) or self._delta_vecs is not None

    def _topk_scratch(self, k: int):
        """Per-thread reusable (oi, od, oi_ptr, od_ptr) for top-k output — no per-query allocation."""
        t = self._tls
        if getattr(t, "k", None) != k:
            t.k = k
            t.oi = np.empty(k, np.int32); t.od = np.empty(k, np.float32)
            t.oip = t.oi.ctypes.data_as(ctypes.c_void_p); t.odp = t.od.ctypes.data_as(ctypes.c_void_p)
        return t.oi, t.od, t.oip, t.odp

    @staticmethod
    def _qptr(query):
        q = query if (query.dtype == np.float32 and query.flags["C_CONTIGUOUS"]) else np.ascontiguousarray(query, np.float32)
        return q.ctypes.data_as(ctypes.c_void_p)

    def _slice_ivf_topk(self, key, start: int, query: np.ndarray, k: int):
        """Top-k over a partition's IVF sublayout (ame_ivf_search over its centroids + sub-offsets).
        Returns (phys_offsets, scores). Scores are exact f32 within probed clusters (rerank-from-f32 inherent)."""
        e = self.pivf[_py(key)]
        cents, offs, nl, npb = e["cents"], e["offsets"], e["nlist"], e["nprobe"]
        oi, od, oip, odp = self._topk_scratch(k)
        self.lib.ame_ivf_search(
            cents.ctypes.data_as(ctypes.c_void_p), nl,
            ctypes.c_void_p(self._bp.value + start * self.dim * 4), offs.ctypes.data_as(ctypes.c_void_p), self.dim,
            self._qptr(query), npb, k, oip, odp)
        v = oi >= 0
        return oi[v].astype(np.int64) + start, od[v]

    def _partition_scan(self, key, ent, query, kf, scan):
        if scan == "ivf" and _py(key) in self.pivf:
            if self.compact_primary == "pq":
                return self._slice_ivf_pq(key, ent[0], query, kf)
            if not self.compact_primary:
                return self._slice_ivf_topk(key, ent[0], query, kf)
            # compact-i8 has no f32/PQ IVF scan -> flat i8 slice scan (correct, just not sub-linear)
        return self._slice_topk(ent[0], ent[1], query, min(kf, ent[1]), scan)

    def _slice_ivf_pq(self, key, start: int, query: np.ndarray, k: int):
        """PQ-IVF: probe centroids, PQ-ADC-scan only those clusters' codes, then i8-rerank -> top-k. Sub-linear
        over a giant partition. nprobe per query is the FIXED calibrated value by default, or PER-QUERY ADAPTIVE
        via TELYS_PQ_NPROBE: 'margin' (probe clusters within a cosine-DELTA of the best centroid — more on
        spread/hard queries, fewer on tight/easy ones) or 'mc' (grow the probe set until the top-k stabilizes).
        Returns (phys_offsets, scores)."""
        e = self.pivf[_py(key)]
        cents, offs, calib = e["cents"], np.asarray(e["offsets"], np.int64), int(e["nprobe"])
        qv = np.ascontiguousarray(query, np.float32); qp = self._qptr(query)
        cs = cents @ qv
        order = np.argsort(-cs)                                         # clusters by centroid score (closest first)
        nlist = len(cents)
        mode = (os.environ.get("TELYS_PQ_NPROBE") or "fixed").lower()
        if mode == "margin":
            try:
                delta = float(os.environ.get("TELYS_PQ_NPROBE_DELTA", "0.05"))
            except ValueError:
                delta = 0.05
            npb = int(np.clip(int(np.sum(cs >= cs[order[0]] - delta)),
                              max(1, calib // 2), min(nlist, max(calib * 4, 64))))
            return self._pq_ivf_pool(order[:npb], offs, start, qp, k)
        if mode == "mc":
            return self._pq_ivf_mc(order, offs, start, qp, k, calib, nlist)
        return self._pq_ivf_pool(order[:calib], offs, start, qp, k)     # fixed (default; unchanged)

    def _pq_ivf_pool(self, probe, offs, start: int, qp, k: int):
        """Gather the probed clusters' PQ codes -> ADC scan -> i8 rerank -> top-k (phys_offsets, scores)."""
        self._last_nprobe = int(len(probe))    # instrumentation: clusters probed on this query (avg-cost metric)
        pos = np.concatenate([np.arange(offs[c], offs[c + 1], dtype=np.int64) for c in probe]) \
            if len(probe) else np.empty(0, np.int64)
        if len(pos) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        m, ks = self._pq_m, self._pq_ksub
        g_codes = np.ascontiguousarray(self._pq_codes.reshape(-1, m)[start + pos])
        pool = int(min(max(self.cand_cap or 0, 8 * k), len(g_codes)))
        lut = np.zeros(m * ks, np.float32)
        pid = np.full(pool, -1, np.int32); pds = np.zeros(pool, np.float32)
        self.lib.ame_flat_topk_pq(g_codes.ctypes.data_as(ctypes.c_void_p), len(g_codes), m, ks, self._pq_dsub,
                                  self._pq_codebook.ctypes.data_as(ctypes.c_void_p), qp,
                                  lut.ctypes.data_as(ctypes.c_void_p), pool,
                                  pid.ctypes.data_as(ctypes.c_void_p), pds.ctypes.data_as(ctypes.c_void_p))
        sel = pid[pid >= 0]
        if len(sel) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        glob = (start + pos[sel]).astype(np.int64)
        if getattr(self, "_base_q8_2d", None) is not None:             # i8 rerank for precision
            gi8 = np.ascontiguousarray(self._base_q8_2d[glob]); kk = min(k, len(gi8))
            ri = np.full(kk, -1, np.int32); rd = np.zeros(kk, np.float32); qq8 = np.empty(self.dim, np.int8)
            self.lib.ame_flat_topk_i8(gi8.ctypes.data_as(ctypes.c_void_p), len(gi8), self.dim, qp,
                                      qq8.ctypes.data_as(ctypes.c_void_p), kk,
                                      ri.ctypes.data_as(ctypes.c_void_p), rd.ctypes.data_as(ctypes.c_void_p))
            v = ri >= 0
            return glob[ri[v].astype(np.int64)], rd[v]
        o = np.argsort(-pds[:len(sel)])[:k]
        return glob[o], pds[o]

    def _pq_ivf_mc(self, order, offs, start: int, qp, k: int, calib: int, nlist: int):
        """Adaptive nprobe by STABILITY (Monte-Carlo-style): grow the probe set (x2/round) until the top-k stops
        changing (Jaccard >= STAB) for `patience` rounds, capped at MAX. Spends probe budget only until the
        answer settles -> fewer clusters on easy queries, more on hard ones."""
        try:
            maxnp = int(os.environ.get("TELYS_PQ_NPROBE_MAX", str(min(nlist, max(calib * 4, 64)))))
            patience = int(os.environ.get("TELYS_PQ_NPROBE_PATIENCE", "1"))
            stab = float(os.environ.get("TELYS_PQ_NPROBE_STAB", "0.9"))
        except ValueError:
            maxnp, patience, stab = min(nlist, max(calib * 4, 64)), 1, 0.9
        npb = max(1, calib // 2)
        prev = None; ok = 0; res = (np.empty(0, np.int64), np.empty(0, np.float32))
        while True:
            res = self._pq_ivf_pool(order[:npb], offs, start, qp, k)
            cur = frozenset(int(i) for i in res[0])
            if prev is not None and cur and (len(cur & prev) / max(1, len(cur | prev))) >= stab:
                ok += 1
                if ok >= patience:
                    break
            else:
                ok = 0
            prev = cur
            if npb >= maxnp:
                break
            npb = min(maxnp, npb * 2)
        return res

    def _resolve_partition(self, query, k: int, key, ent, scan_dtype: str, S):
        """Score base slice (exact/fp16/int8/IVF) + delta-for-key; keep each id's latest VISIBLE version here."""
        base_rows = ent[1] if ent else 0
        if not self._mutated():
            if ent is None:
                return np.empty(0, np.int64), np.empty(0, np.float32), 0
            phys, sc = self._partition_scan(key, ent, query, k, scan_dtype)
            return self.phys_rowid[phys], sc, base_rows
        kf = k + self._margin(key)                         # overfetch past tombstoned/superseded candidates
        if ent is not None and base_rows > 0:
            phys, b_sc = self._partition_scan(key, ent, query, kf, scan_dtype)
            b_ids = self.phys_rowid[phys]; b_lsn = np.zeros(len(b_ids), np.int64)
        else:
            b_ids, b_sc, b_lsn = np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.int64)
        d_ids, d_sc, d_lsn = self._delta_cands(key, query, kf, S)
        cid = np.concatenate([b_ids, d_ids]); csc = np.concatenate([b_sc, d_sc]); cl = np.concatenate([b_lsn, d_lsn])
        keep_i, keep_s = [], []
        for j in np.argsort(-csc):
            lid = int(cid[j])
            lat_lsn, lat_key = self._visible_key(lid, S)
            if lat_lsn == int(cl[j]) and lat_key == _py(key):   # the candidate IS the latest visible version here
                keep_i.append(lid); keep_s.append(float(csc[j]))
                if len(keep_i) >= k:
                    break
        return np.array(keep_i, np.int64), np.array(keep_s, np.float32), base_rows

    # ── mutation (append-only delta with per-key mini-partitions) ──────────────────────────────────

    # ── stats ──────────────────────────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        if getattr(self, "base", None) is None:           # never-built/never-sealed OR compact-primary (i8-only)
            bq8 = getattr(self, "base_q8", None)
            if bq8 is not None and getattr(self, "n", 0):  # compact-primary serve artifact: size from the i8 slab
                b = bq8.nbytes + self.phys_rowid.nbytes + self._log2phys.nbytes
                return {"n": self.n, "dim": self.dim, "dtype": self.dtype, "partitions": len(self.pdir),
                        "build_ms": 0.0, "index_mb": b / 1e6, "compact_primary": True,
                        "overhead_vs_raw_vectors": b / (self.n * self.dim * 4) if self.n else 0.0, "delta_rows": 0,
                        "embedding_space": self.profile.space_id() if self.profile is not None else None}
            return {"n": getattr(self, "n", 0), "dim": self.dim, "dtype": self.dtype,
                    "partitions": len(getattr(self, "pdir", {}) or {}), "build_ms": 0.0, "index_mb": 0.0,
                    "overhead_vs_raw_vectors": 0.0, "delta_rows": 0,
                    "embedding_space": self.profile.space_id() if getattr(self, "profile", None) is not None else None}
        b = self.base.nbytes + self.phys_rowid.nbytes + self._log2phys.nbytes
        shadow = getattr(self, "base_f16", getattr(self, "base_q8", None))
        if shadow is not None:
            b += shadow.nbytes
        raw = self.n * self.dim * 4
        return {
            "n": self.n, "dim": self.dim, "dtype": self.dtype, "partitions": len(self.pdir),
            "build_ms": self.build_ns / 1e6, "index_mb": b / 1e6,
            "overhead_vs_raw_vectors": b / raw if raw else 0.0,
            "delta_rows": 0 if self._delta_vecs is None else len(self._delta_rowids),
            "embedding_space": self.profile.space_id() if self.profile is not None else None,
        }

    # ── persistence: seal to disk + reopen without rebuild (mmap; fail-closed on version/checksum) ──
    FORMAT = "ame.partitioned"
    VERSION = 1

    def _recalibrate_pq_nprobe(self, codes, cb_flat, M, target: float = 0.90, samples: int = 64,
                               k: int = 10, max_np: int = 64) -> None:
        """Re-pick each IVF partition's nprobe so the ACTUAL PQ-IVF pipeline (cluster-probe -> PQ ADC ->
        i8 rerank) hits `target` recall@k vs exact f32. The f32-calibrated nprobe under-probes for PQ because
        IVF-miss and PQ-quantization losses compound. Numpy sim (matches the kernel path); runs at save time."""
        ksub, dsub = 256, self.dim // M
        cb = cb_flat.reshape(M, ksub, dsub)
        b_q8 = self.base_q8.reshape(self.n, self.dim)
        rng = np.random.default_rng(0)
        for key, e in self.pivf.items():
            start, L = self.pdir[key]
            if L <= 0:
                continue
            cents, offs = e["cents"], np.asarray(e["offsets"], np.int64)
            bf = self.base[start:start + L]; cd = codes[start:start + L]; bi = b_q8[start:start + L]
            grid = [p for p in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64) if p <= min(max_np, len(cents))]
            qs = []
            for _ in range(samples):
                i = int(rng.integers(0, L))
                q = bf[i] + 0.2 * rng.standard_normal(self.dim).astype(np.float32)
                n = np.linalg.norm(q); qs.append((q / n).astype(np.float32) if n else bf[i])
            chosen = grid[-1]
            for npb in grid:
                hit = 0
                for q in qs:
                    truth = set(np.argsort(-(bf @ q))[:k].tolist())
                    probe = np.argsort(-(cents @ q))[:npb]
                    pos = np.concatenate([np.arange(offs[c], offs[c + 1]) for c in probe]) if len(probe) else np.empty(0, int)
                    if len(pos) == 0:
                        continue
                    lut = np.stack([q[m * dsub:(m + 1) * dsub] @ cb[m].T for m in range(M)])   # (M, ksub)
                    adc = lut[np.arange(M), cd[pos]].sum(1)                                     # (P,)
                    pool = pos[np.argsort(-adc)[:max(64, 8 * k)]]
                    q8 = np.round(q * 127).clip(-127, 127).astype(np.int32)
                    top = pool[np.argsort(-(bi[pool].astype(np.int32) @ q8))[:k]]
                    hit += len(truth & set(top.tolist()))
                if hit / (samples * k) >= target:
                    chosen = npb
                    break
            e["nprobe"] = int(chosen)

    @staticmethod
    def _pq_choose_m(dim: int) -> int:
        """PQ subquantizer count ~= dim/8 (1 byte each -> ~32x vs f32), constrained to divide dim."""
        m = max(1, dim // 8)
        while dim % m:
            m -= 1
        return m

    def save(self, path: str, compact=False, ivf_min_rows: int = 0) -> str:
        """Seal the index to a directory: vector slab(s) + remap + directory + columns + delta + manifest.

        Manifest carries format/version, dtype/dim, key_name, embedding profile, the measured crossover, and
        a per-file sha256. Reopen via PartitionedVectorIndex.open(path) — mmaps the slabs, no rebuild.

        compact writes an ULTRA-SMALL, READ-ONLY serve artifact, dropping the f32 base:
          - True / "int8": the int8 slab only (4x smaller on disk, ~1.8x faster scan).
          - "pq": Product-Quantization codes (~32x SCAN working set) + the int8 slab for rerank (the
            super-large-repo tier). The PQ codebook is trained here OFFLINE by faiss (like the IVF kmeans);
            the query-time ADC scan stays pure Mojo. Reopens into compact_primary mode (PQ-scan -> i8-rerank).

        ivf_min_rows>0 (OPT-IN, default OFF, compact="pq" only): give partitions >= ivf_min_rows a sub-linear
        PQ-IVF sublayout (cluster-probe -> PQ ADC -> i8 rerank), with nprobe re-calibrated for PQ. This is a
        SPEED/RECALL TRADEOFF — measured ~2.2x faster at 100k but recall is capped by IVF cluster-miss + PQ
        loss compounding (validate on your data). Default (ivf_min_rows=0) keeps the flat PQ scan (higher
        recall, O(n) but cache-friendly). Requires dtype="int8" and a folded index (no pending delta).
        Mutations/off-key filters need the full-precision index. Precision is restored by the i8 rerank.
        """
        mode = "int8" if compact is True else (compact or None)   # -> None | "int8" | "pq"
        if mode:
            if mode not in ("int8", "pq"):
                raise ValueError(f"unknown compact mode {mode!r} (use True/'int8' or 'pq')")
            if self.dtype != "int8" or not hasattr(self, "base_q8"):
                raise ValueError("compact save requires dtype='int8' (an int8 base slab to keep)")
            if self._delta_vecs is not None:
                raise ValueError("compact save requires a folded index — call compact() first (no pending delta)")
            if mode == "pq" and self.n < 256:
                raise ValueError("compact='pq' needs >=256 rows to train the PQ codebook; use 'int8' for tiny indexes")
            # OPT-IN IVF (ivf_min_rows>0): give oversized partitions a sub-linear PQ-IVF sublayout BEFORE
            # PQ-coding (build_partition_ivf reorders base/i8/rowids by cluster; PQ codes are then built on the
            # reordered base, so codes + cluster offsets stay consistent). PQ-IVF query = cluster-probe ->
            # PQ ADC scan of probed clusters -> i8 rerank (nprobe re-calibrated for PQ). A speed/recall
            # tradeoff (default OFF -> flat PQ). 'pq' only (the i8 flat scan ignores clusters).
            if mode == "pq" and not self.pivf and ivf_min_rows > 0 \
                    and any(l >= ivf_min_rows for _, (s, l) in self.pdir.items()):
                self.build_partition_ivf(min_rows=ivf_min_rows)
        os.makedirs(path, exist_ok=True)
        files: dict = {}

        # Atomic per-file publish: write to a temp file in the SAME dir, then os.replace() (atomic rename on
        # POSIX/Windows). A concurrent reader (e.g. another engine reopening while a periodic-save thread runs)
        # then sees either the complete old file or the complete new one — never a torn/partial file (which
        # previously surfaced as `EOFError: No data left in file` from np.load of a col_*.npy mid-write).
        def w(name, arr):
            p = os.path.join(path, name)
            tmp = p + ".tmp"
            np.ascontiguousarray(arr).tofile(tmp)
            os.replace(tmp, p)
            files[name] = {"sha256": _sha256_file(p), "bytes": os.path.getsize(p)}

        if not mode:                                       # compact (int8/pq) drops the f32 base
            w("vectors.f32", self.base.astype(np.float32, copy=False))
        w("rowids.i64", self.phys_rowid.astype(np.int64, copy=False))
        if hasattr(self, "base_f16"):
            w("vectors.f16", self.base_f16)
        if hasattr(self, "base_q8"):                       # kept in int8 AND pq compact (pq reranks on it)
            w("vectors.i8", self.base_q8)
        pq_meta = None
        if mode == "pq":                                   # train the PQ codebook offline (faiss) + write codes
            import faiss as _f
            base_f32 = np.ascontiguousarray(self.base, np.float32)
            M = self._pq_choose_m(self.dim)
            pq = _f.ProductQuantizer(self.dim, M, 8)
            pq.train(base_f32)
            codes = pq.compute_codes(base_f32)             # (n, M) uint8
            cb_flat = _f.vector_to_array(pq.centroids).astype(np.float32)
            w("vectors.pq", codes)
            w("pq_codebook.f32", cb_flat)
            pq_meta = {"m": int(M), "ksub": 256, "dsub": int(self.dim // M)}
            if self.pivf:                                  # PQ-aware nprobe: f32-calibrated nprobe under-probes
                self._recalibrate_pq_nprobe(codes, cb_flat, M)   # for PQ (two lossy stages compound)
        for c in self.columns:
            name = f"col_{c}.npy"; p = os.path.join(path, name); tmp = p + ".tmp"
            with open(tmp, "wb") as _fh:           # np.save(handle) keeps the exact name (no .npy re-append)
                np.save(_fh, self.columns[c])
            os.replace(tmp, p)
            files[name] = {"sha256": _sha256_file(p), "bytes": os.path.getsize(p)}
        has_delta = self._delta_vecs is not None
        if has_delta:
            w("delta.f32", self._delta_vecs.astype(np.float32, copy=False))
            w("delta_rowids.i64", self._delta_rowids.astype(np.int64, copy=False))
            w("delta_lsn.i64", self._delta_lsn.astype(np.int64, copy=False))
        pivf_meta = []
        if self.pivf:
            pk = list(self.pivf.keys())
            w("pivf_cents.f32", np.concatenate([self.pivf[k]["cents"].reshape(-1) for k in pk]).astype(np.float32))
            w("pivf_offs.i32", np.concatenate([self.pivf[k]["offsets"] for k in pk]).astype(np.int32))
            pivf_meta = [[_py(k), int(self.pivf[k]["nlist"]), int(self.pivf[k]["nprobe"])] for k in pk]
        manifest = {
            "format": self.FORMAT, "version": self.VERSION, "dim": int(self.dim), "dtype": self.dtype,
            "n": int(self.n), "key_name": self.key_name, "metric": "ip", "cand_cap": self.cand_cap,
            "partitions": len(self.pdir), "exact_crossover": int(self.exact_crossover),
            "directory": [[_py(k), int(s), int(l)] for k, (s, l) in self.pdir.items()],
            "columns": list(self.columns),
            "profile": self.profile.as_dict() if self.profile is not None else None,
            "has_delta": has_delta,
            "delta_keys": [_py(k) for k in self._delta_keys] if has_delta else [],
            "lsn": int(self._lsn),                                       # MVCC watermark
            "tomb": [[int(i), int(l)] for i, l in self._tomb.items()],   # tombstones (id -> delete LSN)
            "dirty": [[_py(k), int(c)] for k, c in self._dirty.items()],
            "compact_primary": mode or False,    # False | "int8" | "pq" — compressed serve artifact (no f32)
            "pq": pq_meta,                        # PQ params (m/ksub/dsub) when compact_primary == "pq"
            "pivf": pivf_meta,                                           # per-partition IVF (key, nlist, nprobe)
            "files": files,
        }
        # Manifest is the commit point — write it atomically and LAST, so a reader never sees a manifest that
        # references files not yet fully written.
        mtmp = os.path.join(path, "manifest.json.tmp")
        with open(mtmp, "w") as f:
            json.dump(manifest, f)
        os.replace(mtmp, os.path.join(path, "manifest.json"))
        return path

    @classmethod
    def open(cls, path: str, verify: bool = True) -> "PartitionedVectorIndex":
        """Reopen a sealed index by mmap (no rebuild). Fail-closed on unknown version or checksum mismatch."""
        with open(os.path.join(path, "manifest.json")) as f:
            m = json.load(f)
        if m.get("format") != cls.FORMAT or m.get("version") != cls.VERSION:
            raise ValueError(f"unsupported format {m.get('format')!r} v{m.get('version')} "
                             f"(expected {cls.FORMAT!r} v{cls.VERSION})")
        if verify:
            for name, meta in m["files"].items():
                if _sha256_file(os.path.join(path, name)) != meta["sha256"]:
                    raise ValueError(f"checksum mismatch on {name} — corrupt or tampered (fail-closed)")
        self = cls(m["dim"], m["dtype"], cand_cap=m.get("cand_cap"))
        n, d = m["n"], m["dim"]
        cp = m.get("compact_primary", False)
        self.compact_primary = "int8" if cp is True else (cp or False)   # back-compat: #46 wrote bool True
        if self.compact_primary:               # compressed serve artifact (int8/pq): no f32 base on disk
            self.base = None
            self._bp = None
        elif n == 0:                           # empty collection: vectors.f32 is 0 bytes (cannot mmap an empty file)
            self.base = np.empty((0, d), np.float32)
            self._bp = self.base.ctypes.data_as(ctypes.c_void_p)
        else:
            self.base = np.memmap(os.path.join(path, "vectors.f32"), dtype=np.float32, mode="r", shape=(n, d))
            self._bp = self.base.ctypes.data_as(ctypes.c_void_p)
        self.phys_rowid = np.fromfile(os.path.join(path, "rowids.i64"), dtype=np.int64)
        self._log2phys = np.empty(int(self.phys_rowid.max()) + 1, np.int64) if n else np.empty(0, np.int64)
        self._log2phys[self.phys_rowid] = np.arange(n, dtype=np.int64)
        self.pdir = {_py(k): (int(s), int(l)) for k, s, l in m["directory"]}
        self.n, self.key_name = n, m["key_name"]
        self.exact_crossover = m.get("exact_crossover", 1 << 62)
        self.build_ns = 0  # reopened, not built this session
        # reconstruct MVCC state: per-base-row key (from directory), LSN watermark, tombstones, dirty counts
        self._base_key = np.empty(n, dtype=object)
        for k, s, l in m["directory"]:
            self._base_key[int(s):int(s) + int(l)] = _py(k)
        self._lsn = int(m.get("lsn", 0))
        self._tomb = {int(i): int(l) for i, l in m.get("tomb", [])}
        self._dirty = {_py(k): int(c) for k, c in m.get("dirty", [])}
        if n and "vectors.f16" in m["files"]:   # n==0 -> 0-byte shadow, cannot mmap; the empty base suffices
            self.base_f16 = np.memmap(os.path.join(path, "vectors.f16"), dtype=np.float16, mode="r", shape=(n * d,))
            self._bq = self.base_f16.ctypes.data_as(ctypes.c_void_p)
        if n and "vectors.i8" in m["files"]:
            self.base_q8 = np.memmap(os.path.join(path, "vectors.i8"), dtype=np.int8, mode="r", shape=(n * d,))
            self._bq = self.base_q8.ctypes.data_as(ctypes.c_void_p)
        if self.compact_primary == "pq" and n:   # PQ codes + offline-trained codebook (query-time ADC stays Mojo)
            pm = m.get("pq") or {}
            self._pq_m, self._pq_ksub, self._pq_dsub = int(pm["m"]), int(pm["ksub"]), int(pm["dsub"])
            self._pq_codes = np.memmap(os.path.join(path, "vectors.pq"), dtype=np.uint8, mode="r", shape=(n * self._pq_m,))
            self._pq_codebook = np.ascontiguousarray(np.fromfile(os.path.join(path, "pq_codebook.f32"), dtype=np.float32))
            self._base_q8_2d = np.asarray(self.base_q8).reshape(n, d) if hasattr(self, "base_q8") else None
        self.columns = {c: np.load(os.path.join(path, f"col_{c}.npy"), allow_pickle=True) for c in m.get("columns", [])}
        if m.get("profile"):
            from memengine.embedding import EmbeddingProfile
            pf = {k: v for k, v in m["profile"].items() if k != "space_id"}
            self.profile = EmbeddingProfile(**pf)
        if m.get("has_delta"):
            self._delta_vecs = np.fromfile(os.path.join(path, "delta.f32"), dtype=np.float32).reshape(-1, d)
            self._delta_rowids = np.fromfile(os.path.join(path, "delta_rowids.i64"), dtype=np.int64)
            self._delta_lsn = np.fromfile(os.path.join(path, "delta_lsn.i64"), dtype=np.int64)
            self._delta_keys = np.array(m["delta_keys"])
            self._delta_map = {}
            for i, key in enumerate(self._delta_keys):
                self._delta_map.setdefault(_py(key), []).append(i)
                lid = int(self._delta_rowids[i])             # rebuild latest-version-per-id (current view)
                if lid not in self._delta_latest or self._delta_lsn[i] >= self._delta_latest[lid][0]:
                    self._delta_latest[lid] = (int(self._delta_lsn[i]), _py(key), i)
        if m.get("pivf"):                                    # reconstruct per-partition IVF sublayouts
            cents_all = np.fromfile(os.path.join(path, "pivf_cents.f32"), dtype=np.float32)
            offs_all = np.fromfile(os.path.join(path, "pivf_offs.i32"), dtype=np.int32)
            ci = oi = 0
            for key, nl, npb in m["pivf"]:
                c = np.ascontiguousarray(cents_all[ci:ci + nl * d].reshape(nl, d)); ci += nl * d
                o = np.ascontiguousarray(offs_all[oi:oi + nl + 1]); oi += nl + 1
                self.pivf[_py(key)] = {"cents": c, "offsets": o, "nlist": int(nl), "nprobe": int(npb)}
        self._sealed = True
        return self
