"""NativeRuntime — ctypes shim over the native Telys runtime (Phase 1b).

Implements the full `telys.runtime.RuntimeHandle` against `libty_runtime.dylib` (built from
mojo/engine/ty_runtime.mojo): native sealed base + partition directory + exact search + IVF rerank + scatter
+ persistence export, with the delta/MVCC/columns/compaction layer host-orchestrated for now (see
internal/PHASE-1B-NATIVE-RUNTIME.md). At functional parity with LocalRuntime (the runtime-agnostic parity suite
passes under --runtime native). Concurrency: a per-index writer-preferring RWLock (`@_read`/`@_write`) gives
parallel readers + an exclusive (reentrant) writer over the host-side dicts.

Opaque handles are `c_void_p` the SDK never introspects; `_NativeIndex` owns the handle and frees it on GC.
"""
from __future__ import annotations

import ctypes
import functools
import os
import sys
import threading
from contextlib import contextmanager

from telys.runtime import FORMAT_VERSION as _SDK_FORMAT_VERSION
from telys.runtime import RuntimeHandle
from memengine import _validate

_DTYPE_CODE = {"f32": 0, "fp16": 1, "int8": 2}
_CODE_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}


class _RWLock:
    """Writer-preferring readers-writer lock: parallel readers, exclusive writer. A WAITING writer blocks
    new readers so in-flight readers drain (no writer starvation). The WRITE side is reentrant per-thread
    (so a writer can call another @_write method, e.g. build_partition_ivf -> compact -> build); a READ
    performed inside a write must use the UNLOCKED `_stats` (a @_read inside a write would self-deadlock)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0
        self._wthread = None       # owning writer thread (write lock is REENTRANT per thread)
        self._wdepth = 0

    @contextmanager
    def read(self):
        with self._cond:
            while self._writer or self._writers_waiting:
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
        me = threading.get_ident()
        with self._cond:
            if self._writer and self._wthread == me:    # reentrant: same thread already holds write
                self._wdepth += 1
            else:
                self._writers_waiting += 1
                while self._writer or self._readers:
                    self._cond.wait()
                self._writers_waiting -= 1
                self._writer = True
                self._wthread = me
                self._wdepth = 1
        try:
            yield
        finally:
            with self._cond:
                self._wdepth -= 1
                if self._wdepth == 0:
                    self._writer = False
                    self._wthread = None
                    self._cond.notify_all()


def _read(fn):
    """Guard a read-only RuntimeHandle method under the index read lock (parallel readers)."""
    @functools.wraps(fn)
    def w(self, index, *a, **k):
        with index._rw.read():
            return fn(self, index, *a, **k)
    return w


def _write(fn):
    """Guard a mutating RuntimeHandle method under the index write lock (exclusive)."""
    @functools.wraps(fn)
    def w(self, index, *a, **k):
        with index._rw.write():
            return fn(self, index, *a, **k)
    return w


def _coerce_where(where):
    """Normalize a where filter to (column, value); (None, None) for full-scan or unsupported shapes."""
    if where is None:
        return None, None
    if isinstance(where, dict):
        if len(where) != 1:
            return None, None
        (c, v), = where.items()
        return c, v
    col = getattr(where, "column", None)   # telys.filters.Eq
    if col is not None:
        return col, getattr(where, "value", None)
    return None, None
_LIB = None
_LIB_PATH = None


def _lib_ext() -> str:
    """Shared-library extension for this platform — mirrors telys.paths.lib_ext() but standalone,
    so native-lib resolution works even where importing telys.paths would be awkward."""
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform.startswith("win"):
        return "dll"
    return "so"


def _installed_lib(stem: str):
    """Path to a verified `telys runtime install`ed lib (via telys.paths), or None. Stdlib-only; never raises."""
    try:
        from telys.paths import installed_lib_path
        return installed_lib_path(stem)
    except Exception:  # noqa: BLE001 — resolution must never break engine load
        return None


def _under_install_cache(path) -> bool:
    """True if `path` is inside the verified `telys runtime install` cache ($TELYS_HOME/runtime/...)."""
    try:
        from telys.paths import telys_home
        return bool(path) and os.path.abspath(path).startswith(os.path.join(telys_home(), "runtime") + os.sep)
    except Exception:  # noqa: BLE001
        return False


def _resolve_native_lib() -> str:
    p = os.environ.get("TELYS_NATIVE_KERNEL") or os.environ.get("AME_NATIVE_KERNEL")
    if p:
        return p
    installed = _installed_lib("libty_runtime")   # verified `telys runtime install` location (zero config)
    if installed:
        return installed
    ext = _lib_ext()
    here = os.path.dirname(__file__)
    bundled = os.path.join(here, "_runtime", f"libty_runtime.{ext}")   # internal full-engine wheel
    if os.path.exists(bundled):
        return bundled
    return os.path.abspath(os.path.join(here, "..", "..", "..", "mojo_build", f"libty_runtime.{ext}"))


def _lib():
    global _LIB, _LIB_PATH
    if _LIB is None:
        _LIB_PATH = _resolve_native_lib()
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"Telys native runtime not found at {_LIB_PATH}. Build it: cd mojo_env && pixi run mojo "
                "build --emit shared-lib ../mojo/engine/ty_runtime.mojo "
                f"-o ../mojo_build/libty_runtime.{_lib_ext()} "
                "(or set TELYS_NATIVE_KERNEL).")
        lib = ctypes.CDLL(_LIB_PATH)
        lib.ty_runtime_format_version.restype = ctypes.c_int32
        lib.ty_index_new.argtypes = [ctypes.c_int32, ctypes.c_int32]
        lib.ty_index_new.restype = ctypes.c_void_p
        lib.ty_index_sealed.argtypes = [ctypes.c_void_p]
        lib.ty_index_sealed.restype = ctypes.c_int32
        lib.ty_index_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                                       ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                                       ctypes.POINTER(ctypes.c_int32)]
        lib.ty_index_stats.restype = ctypes.c_int32
        lib.ty_index_free.argtypes = [ctypes.c_void_p]
        lib.ty_index_free.restype = None
        lib.ty_index_build.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int32]
        lib.ty_index_build.restype = ctypes.c_int32
        lib.ty_index_search_partition.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                                  ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_search_partition.restype = ctypes.c_int32
        lib.ty_index_search_full.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                                             ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_search_full.restype = ctypes.c_int32
        lib.ty_index_scatter_topk.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
                                              ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_scatter_topk.restype = ctypes.c_int32
        lib.ty_index_export_base.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_export_base.restype = ctypes.c_int32
        # ── MVCC delta layer (1b-B) ──
        lib.ty_index_snapshot.argtypes = [ctypes.c_void_p]; lib.ty_index_snapshot.restype = ctypes.c_int64
        lib.ty_index_mutated.argtypes = [ctypes.c_void_p]; lib.ty_index_mutated.restype = ctypes.c_int32
        lib.ty_index_delta_count.argtypes = [ctypes.c_void_p]; lib.ty_index_delta_count.restype = ctypes.c_int64
        lib.ty_index_visible_count.argtypes = [ctypes.c_void_p]; lib.ty_index_visible_count.restype = ctypes.c_int64
        lib.ty_index_visible_ids.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_visible_ids.restype = ctypes.c_int64
        lib.ty_index_insert.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.ty_index_insert.restype = ctypes.c_int64
        lib.ty_index_delete.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.ty_index_delete.restype = ctypes.c_int64
        lib.ty_index_export_visible.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_export_visible.restype = ctypes.c_int64
        lib.ty_index_search_mvcc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
                                             ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_search_mvcc.restype = ctypes.c_int32
        # ── filter metadata (1b-B B3) ──
        lib.ty_index_set_rowmeta.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_int32, ctypes.c_int32]
        lib.ty_index_set_rowmeta.restype = ctypes.c_int32
        lib.ty_index_search_filtered.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64,
                                                 ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_search_filtered.restype = ctypes.c_int32
        lib.ty_index_match_ids.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64,
                                           ctypes.c_void_p]
        lib.ty_index_match_ids.restype = ctypes.c_int64
        lib.ty_index_export_rowmeta.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                                                ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_export_rowmeta.restype = ctypes.c_int32
        # ── IVF directory (1b-B B4) ──
        lib.ty_index_set_ivf.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.ty_index_set_ivf.restype = ctypes.c_int32
        lib.ty_index_clear_ivf.argtypes = [ctypes.c_void_p]; lib.ty_index_clear_ivf.restype = ctypes.c_int32
        lib.ty_index_has_ivf.argtypes = [ctypes.c_void_p, ctypes.c_int64]; lib.ty_index_has_ivf.restype = ctypes.c_int32
        lib.ty_index_set_ivf_nprobe.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32]
        lib.ty_index_set_ivf_nprobe.restype = ctypes.c_int32
        lib.ty_index_search_ivf.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int32,
                                            ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_search_ivf.restype = ctypes.c_int32
        # ── tuner decision policy (1b-B B5) ──
        lib.ty_tuner_defaults.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; lib.ty_tuner_defaults.restype = ctypes.c_int32
        lib.ty_index_tuner_oversized.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
        lib.ty_index_tuner_oversized.restype = ctypes.c_int32
        fv = int(lib.ty_runtime_format_version())
        if fv != _SDK_FORMAT_VERSION:
            raise RuntimeError(f"native runtime format v{fv} != SDK format v{_SDK_FORMAT_VERSION}")
        _LIB = lib
    return _LIB


class _NativeIndex:
    """Owns an opaque native index handle (c_void_p) and frees it on GC. The SDK treats it as opaque."""

    def __init__(self, handle, dim, dtype):
        self._h = handle
        self.dim = dim
        self.dtype = dtype
        self.key_name = None
        # 1b-B B3: per-row filter metadata (key_id + column value_ids) lives NATIVELY now. The host keeps only
        # the tiny value<->id VOCAB needed to translate user filter values to/from the interned ids the native
        # match uses — not the per-row data. keymap == key_val2id (also the base directory key_id scheme).
        self.keymap: dict = {}      # partition key value -> key_id  (== base directory kid; grows for delta keys)
        self.key_i2v: dict = {}     # key_id -> partition key value (de-intern for materialize)
        self.part_rows: dict = {}   # partition key value -> base row count (for explain)
        self.col_names: list = []   # filter column names, ordered (position == native col index)
        self.col_v2i: dict = {}     # col -> {value -> value_id}
        self.col_i2v: dict = {}     # col -> {value_id -> value}  (de-intern for columns_for/materialize)
        # IVF-within-partition (1b-B B4): the directory (centroids + cluster lists) + probe/gather/rerank are
        # NATIVE now. faiss kmeans + nprobe calibration stay host/build-time (D-26). The host keeps only a tiny
        # membership set + (nlist,nprobe) for routing/explain — not the cluster data.
        self.ivf_keys: set = set()  # partition key VALUES that have a native IVF directory
        self.ivf_meta: dict = {}    # key value -> {"nlist": int, "nprobe": int}  (explain only)
        self.exact_crossover = 1 << 62
        self._rw = _RWLock()        # parallel readers / exclusive writer over the host vocab/IVF + native state

    # ── interning: user filter values <-> compact integer ids the native match uses ──
    def _intern_key(self, value):
        i = self.keymap.get(value)
        if i is None:
            i = len(self.keymap)
            self.keymap[value] = i
            self.key_i2v[i] = value
        return i

    def _intern_col(self, col, value):
        v2i = self.col_v2i.setdefault(col, {})
        i = v2i.get(value)
        if i is None:
            i = len(v2i)
            v2i[value] = i
            self.col_i2v.setdefault(col, {})[i] = value
        return i

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                _lib().ty_index_free(self._h)
                self._h = None
        except Exception:  # noqa: BLE001 — never raise during GC
            pass


class _NativeTuner:
    """Kernel-native default tuner (1b-B B5). The DECISION POLICY + thresholds + constants live in
    libty_runtime (`ty_tuner_defaults` / `ty_index_tuner_oversized`); this is a thin plan-assembly shim that
    emits the public `telys.tuning.TuningPlan`. No readable-Python tuner policy ships in the runtime. Expected
    metrics are informational (measured optionally at apply via faiss, D-26 — never required, never hot-path).
    The plan is APPLIED by the runtime-agnostic facade (set_exact_crossover + native build_partition_ivf)."""
    name = "heuristic"

    def __init__(self, target_recall=None, ivf_min_rows=None):
        mr, tm = ctypes.c_int32(), ctypes.c_int32()
        _lib().ty_tuner_defaults(ctypes.byref(mr), ctypes.byref(tm))   # policy constants from the kernel
        self.ivf_min_rows = int(ivf_min_rows) if ivf_min_rows is not None else int(mr.value)
        self.target_recall = float(target_recall) if target_recall is not None else tm.value / 1000.0

    def tune_collection(self, collection, workload=None, objectives=None):
        import numpy as np
        from telys.tuning import TuningPlan, _hash, _k, _workload_digest
        idx = collection.idx
        cons = (objectives or {}).get("constraints", {}) if objectives else {}
        target = float(cons.get("recall_at_10", cons.get("recall_at_k", self.target_recall)))
        # Read stats + the oversized DECISION + de-intern under ONE read-lock hold: compact() swaps+frees
        # idx._h and rebuilds key_i2v under @_write, so two lock-free native calls would race (use-after-free,
        # and a TOCTOU overflow if `partitions` grows between the stats read and the oversized read).
        with idx._rw.read():
            n, dim, dt, nparts = ctypes.c_int64(), ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
            _lib().ty_index_stats(idx._h, ctypes.byref(n), ctypes.byref(dim), ctypes.byref(dt), ctypes.byref(nparts))
            npart = int(nparts.value)
            crossover = int(getattr(idx, "exact_crossover", 1 << 62))
            oversized = []
            if npart:                              # the IVF-enable DECISION is computed in the kernel
                keyids, rows = np.empty(npart, np.int64), np.empty(npart, np.int64)
                cnt = int(_lib().ty_index_tuner_oversized(idx._h, int(self.ivf_min_rows),
                          keyids.ctypes.data_as(ctypes.c_void_p), rows.ctypes.data_as(ctypes.c_void_p)))
                oversized = sorted(((idx.key_i2v.get(int(keyids[i])), int(rows[i])) for i in range(cnt)),
                                   key=lambda x: -x[1])
        trace = [f"target recall floor = {target}",
                 f"IVF for partitions >= {self.ivf_min_rows} rows (KERNEL policy): {len(oversized)} of {npart}",
                 "per-partition nprobe calibrated to the floor at apply (build-time, faiss; D-26)"]
        if not oversized:
            trace.append("no oversized partitions — exact contiguous slice scans; no IVF")
        return TuningPlan(
            collection=collection.name, tuner=self.name, target_recall=target,
            exact_crossover_rows=crossover,
            ivf={"enabled": bool(oversized), "min_rows": self.ivf_min_rows, "target_recall": target,
                 "partitions": [{"key": _k(k), "rows": r} for k, r in oversized[:64]]},
            constraints=dict(cons), decision_trace=trace,
            workload_hash=_hash(_workload_digest(workload)) if workload is not None else "",
            expected={"basis": "kernel_policy", "recall_at_10_target": target,
                      "note": "IVF nprobe is calibrated to this floor at apply; expected metrics are optional"})

    def choose_plan(self, stats, query):
        return None

    def explain(self):
        return {"tuner": self.name, "policy": "kernel-native (libty_runtime)"}


class NativeRuntime(RuntimeHandle):
    """Native-backed RuntimeHandle (Phase 1b, full surface). Lifecycle + base layout/index/scan/IVF-rerank +
    persistence-export run natively (libty_runtime); MVCC bookkeeping (delta/tombstones/columns/IVF directory)
    is host-orchestrated and guarded by a writer-preferring `_RWLock` (parallel readers / exclusive, reentrant
    writer). The host bookkeeping moves into Mojo in 1b-B before any external runtime release (see
    internal/PHASE-1B-NATIVE-RUNTIME.md)."""

    FORMAT_VERSION = _SDK_FORMAT_VERSION

    # ── lifecycle (native, implemented) ───────────────────────────────────────────────────────────
    def new_index(self, dim, dtype):
        code = _DTYPE_CODE.get(dtype, 0)
        h = _lib().ty_index_new(int(dim), int(code))
        if not h:
            raise RuntimeError("ty_index_new returned NULL")
        return _NativeIndex(h, int(dim), dtype)

    @_read
    def sealed(self, index) -> bool:
        # @_read is REQUIRED: compact() swaps + frees index._h under @_write; reading _h here lock-free
        # would be a use-after-free against a concurrent compaction.
        return bool(_lib().ty_index_sealed(index._h))

    @_read
    def stats(self, index) -> dict:
        return self._stats(index)

    def _stats(self, index) -> dict:
        n, dim = ctypes.c_int64(), ctypes.c_int32()
        dt, parts = ctypes.c_int32(), ctypes.c_int32()
        _lib().ty_index_stats(index._h, ctypes.byref(n), ctypes.byref(dim), ctypes.byref(dt), ctypes.byref(parts))
        visible = int(_lib().ty_index_visible_count(index._h))   # base + delta - tombstoned (native MVCC)
        return {"n": visible, "dim": int(dim.value), "dtype": _CODE_DTYPE.get(int(dt.value), "f32"),
                "partitions": int(parts.value), "build_ms": 0.0, "index_mb": 0.0,
                "overhead_vs_raw_vectors": 0.0, "delta_rows": int(_lib().ty_index_delta_count(index._h)),
                "embedding_space": None, "runtime": "native"}

    def default_tuner(self):
        # Kernel-native policy (B5): the decision/thresholds live in libty_runtime — the shipped runtime needs
        # no readable-Python tuner. (LocalRuntime keeps the reference HeuristicTuner.)
        return _NativeTuner()

    # ── ingest: seal the base into native state (PR-2) ─────────────────────────────────────────────
    @_write
    def build(self, index, vecs, iids, keys, key_name, columns, texts=None):
        import numpy as np                                # texts ignored: lexical index is local-runtime only
        _validate.require_handle(index._h)
        v = _validate.require_finite(np.ascontiguousarray(vecs, np.float32), "base vectors")
        if v.ndim != 2:
            raise ValueError(f"base vectors must be 2-D (n, dim); got shape {tuple(v.shape)}")
        n, dim = v.shape
        _validate.fits_i32(n, "base row count")
        keys = np.asarray(keys)
        order = np.argsort(keys, kind="stable")                 # group rows by partition key (contiguous layout)
        base = np.ascontiguousarray(v[order], np.float32)
        phys = np.ascontiguousarray(np.asarray(iids, np.int64)[order])
        sk = keys[order]
        uniq, starts = np.unique(sk, return_index=True)         # ascending unique keys + slice starts
        ndir = len(uniq)
        dir_key = np.arange(ndir, dtype=np.int64)               # native key_id == position in `uniq`
        dir_start = np.ascontiguousarray(starts.astype(np.int32))
        dir_len = np.ascontiguousarray(np.append(starts[1:], n).astype(np.int32) - starts.astype(np.int32))
        rc = _lib().ty_index_build(
            index._h, base.ctypes.data_as(ctypes.c_void_p), int(n), int(dim),
            phys.ctypes.data_as(ctypes.c_void_p), dir_key.ctypes.data_as(ctypes.c_void_p),
            dir_start.ctypes.data_as(ctypes.c_void_p), dir_len.ctypes.data_as(ctypes.c_void_p), int(ndir))
        if rc != 0:
            raise RuntimeError(f"ty_index_build failed (rc={rc})")
        index.key_name = key_name
        # key_id == base directory kid (position in sorted-unique); the vocab grows for delta-only keys later.
        index.keymap = {(kk.item() if hasattr(kk, "item") else kk): i for i, kk in enumerate(uniq)}
        index.key_i2v = {i: kv for kv, i in index.keymap.items()}
        index.part_rows = {(kk.item() if hasattr(kk, "item") else kk): int(dir_len[i]) for i, kk in enumerate(uniq)}
        index.col_names = list((columns or {}).keys())          # column index == position
        index.col_v2i = {}; index.col_i2v = {}
        # Per-row filter metadata (key_id + column value_ids) is stored NATIVELY (ty_index_set_rowmeta); the
        # host only interns the value<->id vocab. The actual per-row data never lives in readable Python.
        ids_list = [int(x) for x in np.asarray(iids, np.int64)]
        kvals = [(x.item() if hasattr(x, "item") else x) for x in keys]
        self._set_rowmeta(index, ids_list, kvals, columns)

    @staticmethod
    def _set_rowmeta(index, ids, keyvals, columns):
        """Intern (key value, column values) -> ids and push the per-row metadata into native state."""
        import numpy as np
        n = len(ids)
        if n == 0:
            return
        lids = np.ascontiguousarray(np.asarray(ids, np.int64))
        keyids = np.ascontiguousarray(
            np.asarray([index._intern_key(kv.item() if hasattr(kv, "item") else kv) for kv in keyvals], np.int64))
        nc = len(index.col_names)
        colvals = np.full((n, max(nc, 1)), -1, np.int64)
        cmap = columns or {}
        for ci, cname in enumerate(index.col_names):
            if cname in cmap:
                arr = list(cmap[cname])
                for i in range(n):
                    val = arr[i].item() if hasattr(arr[i], "item") else arr[i]
                    colvals[i, ci] = index._intern_col(cname, val)
        colvals = np.ascontiguousarray(colvals)
        _lib().ty_index_set_rowmeta(index._h, lids.ctypes.data_as(ctypes.c_void_p),
                                    keyids.ctypes.data_as(ctypes.c_void_p),
                                    colvals.ctypes.data_as(ctypes.c_void_p), int(nc), int(n))

    # ── query: fast non-mutated key/full paths (directory/full scan + IVF) · native FILTERED match+merge
    #    for off-key and all mutated paths. Filter values are interned to ids; the match runs in Mojo. ──────
    @_read
    def search(self, index, vector, top_k, where, explain, target_recall):
        import numpy as np
        _validate.require_handle(index._h)
        if np.asarray(vector, np.float32).reshape(-1).shape[0] != index.dim:   # n*dim contract (anti-OOB)
            raise ValueError(f"query vector length != index dim {index.dim}")
        col, val = _coerce_where(where)
        value = (val.item() if hasattr(val, "item") else val) if col is not None else None
        # Resolve the filter to (mode, col_idx, value_id). Unknown key/value => matches nothing.
        if col is None:
            mode, ci, vid = 0, -1, 0
        elif col == index.key_name:
            mode, ci, vid = 1, -1, index.keymap.get(value)
        elif col in index.col_names:
            mode, ci, vid = 2, index.col_names.index(col), index.col_v2i.get(col, {}).get(value)
        else:
            raise NotImplementedError(f"NativeRuntime: unknown filter column {col!r} (no usable index/column).")
        # Clamp top-k to [0, visible_count] so the caller-sized output buffer can never be under-allocated and
        # a huge top_k can't trigger a giant alloc; reject a non-finite query (NaN silently corrupts ranking).
        k = _validate.clamp_topk(top_k, max(1, int(_lib().ty_index_visible_count(index._h))))
        q = _validate.require_finite(np.ascontiguousarray(vector, np.float32), "query vector")
        mutated = bool(_lib().ty_index_mutated(index._h))
        plan = "FullScanMask" if mode == 0 else "PartitionSliceExactF32" if mode == 1 else "ScatterGatherExact"
        if col is not None and vid is None:                        # unknown key/column value -> no matches
            ids, sc = np.empty(0, np.int64), np.empty(0, np.float32)
        elif (not mutated) and mode == 1 and target_recall < 1.0 and value in index.ivf_keys:   # IVF approx
            ids, sc = self._ivf_search(index, vid, q, k); plan = "PartitionIVFRerankF32"
        elif (not mutated) and mode == 1:                          # fast exact partition slice (directory)
            ids, sc = self._scan(_lib().ty_index_search_partition, index._h, k, q, c_first=int(vid))
        elif (not mutated) and mode == 0:                          # fast full scan
            ids, sc = self._scan(_lib().ty_index_search_full, index._h, k, q)
        else:                                                      # native filtered match+merge (off-key / mutated)
            ids, sc = self._filtered(index, mode, ci, vid, q, k)
        if explain:
            ex = {"plan": plan, "exact": plan != "PartitionIVFRerankF32", "fallback": mode != 1, "mvcc": mutated,
                  "delta_rows": int(_lib().ty_index_delta_count(index._h)), "candidates_reranked": int(len(ids))}
            if mode == 1:
                ex["partition_key"] = index.key_name; ex["partition_value"] = value
                ex["base_rows"] = index.part_rows.get(value, 0)
                if plan == "PartitionIVFRerankF32":
                    ex["nprobe"] = index.ivf_meta[value]["nprobe"]; ex["nlist"] = index.ivf_meta[value]["nlist"]
            elif mode == 2:
                ex["filter"] = f"{col} == {value!r}"
            return ids, sc, ex
        return ids, sc

    @staticmethod
    def _scan(fn, handle, k, q, c_first=None):
        import numpy as np
        out_ids, out_sc = np.empty(max(k, 1), np.int64), np.empty(max(k, 1), np.float32)
        args = [handle]
        if c_first is not None:
            args.append(c_first)
        args += [q.ctypes.data_as(ctypes.c_void_p), k,
                 out_ids.ctypes.data_as(ctypes.c_void_p), out_sc.ctypes.data_as(ctypes.c_void_p)]
        cnt = _validate.check_count(fn(*args), out_ids.shape[0])   # tripwire: kernel must not write past the buffer
        v = out_ids[:cnt] >= 0
        return out_ids[:cnt][v].copy(), out_sc[:cnt][v].copy()

    @staticmethod
    def _filtered(index, mode, ci, vid, q, k):
        """Native filtered MVCC search: match (key/column id) + skip tombstoned + score delta-or-base + merge."""
        import numpy as np
        out_ids, out_sc = np.empty(max(k, 1), np.int64), np.empty(max(k, 1), np.float32)
        cnt = _lib().ty_index_search_filtered(
            index._h, int(mode), int(ci), int(vid if vid is not None else -1),
            q.ctypes.data_as(ctypes.c_void_p), k,
            out_ids.ctypes.data_as(ctypes.c_void_p), out_sc.ctypes.data_as(ctypes.c_void_p))
        cnt = _validate.check_count(cnt, out_ids.shape[0])         # tripwire: bounded by the caller's buffer
        m = out_ids[:cnt] >= 0
        return out_ids[:cnt][m].copy(), out_sc[:cnt][m].copy()

    @_write
    def insert(self, index, vecs, iids, keys, columns, texts=None):
        # Native: store the delta vectors + bump the LSN + clear tombstones. Host: key/column metadata only.
        # (texts ignored: the lexical index is local-runtime only.)
        import numpy as np
        _validate.require_handle(index._h)
        v = _validate.require_finite(np.ascontiguousarray(vecs, np.float32), "insert vectors")
        _validate.require_matrix(v, index.dim, "insert vectors")    # n*dim contract (anti-OOB)
        ids64 = np.ascontiguousarray(np.asarray(iids, np.int64))
        ids = [int(x) for x in ids64]
        _validate.fits_i32(len(ids), "insert count")
        _lib().ty_index_insert(index._h, v.ctypes.data_as(ctypes.c_void_p),
                               ids64.ctypes.data_as(ctypes.c_void_p), int(len(ids)))
        self._set_rowmeta(index, ids, list(np.asarray(keys)), columns)   # native per-row key_id + col value_ids
        return ids

    @_write
    def update(self, index, iids, vecs, keys, columns, texts=None):
        import numpy as np                                # texts ignored: lexical index is local-runtime only
        _validate.require_handle(index._h)
        v = _validate.require_finite(np.ascontiguousarray(vecs, np.float32), "update vectors")
        _validate.require_matrix(v, index.dim, "update vectors")    # n*dim contract (anti-OOB)
        ids64 = np.ascontiguousarray(np.asarray(iids, np.int64))
        ids = [int(x) for x in ids64]
        _validate.fits_i32(len(ids), "update count")
        _lib().ty_index_insert(index._h, v.ctypes.data_as(ctypes.c_void_p),       # upsert (latest wins) native
                               ids64.ctypes.data_as(ctypes.c_void_p), int(len(ids)))
        if keys is None:                                           # keep current key (de-intern native key_id)
            keyvals = [index.key_i2v.get(kid) for kid in self._row_keyids(index, ids)]
        else:
            keyvals = list(np.asarray(keys))
        self._set_rowmeta(index, ids, keyvals, columns)
        return ids

    @staticmethod
    def _row_keyids(index, ids):
        import numpy as np
        m = len(ids)
        lids = np.ascontiguousarray(np.asarray(ids, np.int64))
        kids = np.empty(max(m, 1), np.int64)
        cols = np.empty(max(m, 1) * max(len(index.col_names), 1), np.int64)
        _lib().ty_index_export_rowmeta(index._h, lids.ctypes.data_as(ctypes.c_void_p), int(m),
                                       kids.ctypes.data_as(ctypes.c_void_p), cols.ctypes.data_as(ctypes.c_void_p))
        return [int(x) for x in kids[:m]]

    @_write
    def delete(self, index, iids):
        import numpy as np
        _validate.require_handle(index._h)
        ids64 = np.ascontiguousarray(np.asarray([int(x) for x in iids], np.int64))
        _validate.fits_i32(len(ids64), "delete count")
        _lib().ty_index_delete(index._h, ids64.ctypes.data_as(ctypes.c_void_p), int(len(ids64)))
        return int(len(ids64))                       # native tombstone hides the row from search/live/visible

    @_read
    def snapshot(self, index):
        return int(_lib().ty_index_snapshot(index._h))

    # ── persistence + compaction: materialization + supersession are NATIVE; host maps key/column metadata ──
    def _materialize_visible(self, index):
        """(vecs[m,dim], ids[m], keys[m], columns{c:[m]}) for the visible set (latest version, tombstoned
        excluded) — computed NATIVELY (ty_index_export_visible); the host only attaches key/column metadata."""
        import numpy as np
        n = int(_lib().ty_index_visible_count(index._h))
        dim = index.dim
        vecs = np.empty(max(n, 1) * dim, np.float32)
        lids = np.empty(max(n, 1), np.int64)
        if n:
            _lib().ty_index_export_visible(index._h, vecs.ctypes.data_as(ctypes.c_void_p),
                                           lids.ctypes.data_as(ctypes.c_void_p))
        V = np.ascontiguousarray(vecs[:n * dim].reshape(n, dim), np.float32) if n else np.empty((0, dim), np.float32)
        ids = [int(x) for x in lids[:n]]
        # de-intern native key_id + column value_ids -> values via the host vocab
        nc = len(index.col_names)
        keyids = np.empty(max(n, 1), np.int64)
        colvals = np.empty(max(n, 1) * max(nc, 1), np.int64)
        vlids = np.ascontiguousarray(lids[:max(n, 1)], np.int64)   # bound contiguous buffer for the C call
        if n:
            _lib().ty_index_export_rowmeta(index._h, vlids.ctypes.data_as(ctypes.c_void_p),
                                           int(n), keyids.ctypes.data_as(ctypes.c_void_p),
                                           colvals.ctypes.data_as(ctypes.c_void_p))
        keys = [index.key_i2v.get(int(keyids[i])) for i in range(n)]
        cv = colvals[:n * nc].reshape(n, nc) if (n and nc) else None
        cols = {cname: [index.col_i2v.get(cname, {}).get(int(cv[i, ci])) for i in range(n)]
                for ci, cname in enumerate(index.col_names)} if (n and nc) else {c: [] for c in index.col_names}
        return V, np.asarray(ids, np.int64), keys, cols

    @_write
    def compact(self, index):
        import numpy as np
        # Parity with LocalRuntime.compact(): return {rows_before, rows_after, reclaimed, lsn} — the
        # RuntimeHandle seam contract (bench/test_engine_api.py asserts it). lsn is 0: the native rebuild
        # seals a FRESH handle whose log starts over (visible_count before/after carries the row accounting).
        before = int(_lib().ty_index_visible_count(index._h))
        if not _lib().ty_index_mutated(index._h):
            return {"rows_before": before, "rows_after": before, "reclaimed": 0, "lsn": 0}
        V, ids, keys, cols = self._materialize_visible(index)     # native export (still on the old handle)
        old = index._h
        index._h = _lib().ty_index_new(int(index.dim), int(_DTYPE_CODE.get(index.dtype, 0)))  # fresh: empty delta/tomb/lsn
        index.keymap, index.key_i2v, index.part_rows = {}, {}, {}
        index.col_names, index.col_v2i, index.col_i2v = [], {}, {}
        index.ivf_keys, index.ivf_meta = set(), {}   # IVF sublayouts invalidated by the re-seal (fresh handle)
        if len(ids):
            self.build(index, V, ids, np.asarray(keys, dtype=object), index.key_name,
                       {c: np.array(v, dtype=object) for c, v in cols.items()})
        _lib().ty_index_free(old)
        after = int(_lib().ty_index_visible_count(index._h))
        return {"rows_before": before, "rows_after": after, "reclaimed": before - after, "lsn": 0}

    @_read
    def save(self, index, path):
        import json
        import numpy as np
        V, ids, keys, cols = self._materialize_visible(index)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "ty_base.npy"), V)
        # Coerce numpy scalars (np.int64/np.float64/np.str_) to plain Python so values round-trip exactly
        # through JSON — otherwise default=str would stringify them and off-key filters miss after reopen.
        sc = lambda x: x.item() if hasattr(x, "item") else x  # noqa: E731
        meta = {"dim": index.dim, "dtype": index.dtype, "key_name": index.key_name,
                "ids": [int(x) for x in ids], "keys": [sc(x) for x in keys],
                "columns": {c: [sc(x) for x in v] for c, v in cols.items()}}
        with open(os.path.join(path, "ty_native.json"), "w") as f:
            json.dump(meta, f)
        return path

    def open_index(self, path):
        import json
        import numpy as np
        base_path = os.path.join(path, "ty_base.npy")
        if not os.path.exists(base_path):
            # fail closed with a clear diagnostic instead of a cryptic FileNotFoundError (D-19): a collection
            # written by a different runtime (e.g. LocalRuntime's manifest.json/vectors.f32) lands here.
            raise RuntimeError(
                f"collection at {path!r} was not written by the native runtime (no ty_base.npy). It may be a "
                "LocalRuntime collection — open it with the matching runtime (TELYS_RUNTIME), or rebuild it.")
        V = np.load(base_path, allow_pickle=False)
        meta = json.load(open(os.path.join(path, "ty_native.json")))
        idx = self.new_index(int(meta["dim"]), meta["dtype"])
        ids = np.asarray(meta["ids"], np.int64)
        if len(ids):
            self.build(idx, np.ascontiguousarray(V, np.float32), ids, np.array(meta["keys"], dtype=object),
                       meta["key_name"], {c: np.array(v, dtype=object) for c, v in meta.get("columns", {}).items()})
        else:
            idx.key_name = meta["key_name"]
        return idx

    # lexical (on-device BM25 code index) — not yet ported to the native runtime; use TELYS_RUNTIME=local.
    # (No @_write: these raise immediately; keeping the decorator on build_partition_ivf below.)
    def build_lexical(self, index, k1=1.8, b=1.0):
        raise NotImplementedError("the lexical (BM25) index is not yet supported on the native runtime; "
                                  "use the local runtime (TELYS_RUNTIME=local) for mode='lexical'")

    def search_lexical(self, index, text, top_k, where, explain):
        raise NotImplementedError("the lexical (BM25) index is not yet supported on the native runtime; "
                                  "use the local runtime (TELYS_RUNTIME=local) for mode='lexical'")

    # compact serve artifacts — the native index handle has no quantized-export path yet; the engine-side
    # sealer (PartitionedVectorIndex.save(compact=...)) lives in the local runtime. Fail closed (D-19 style).
    def export_compact(self, index, path, mode):
        raise NotImplementedError("compact serve artifacts are not yet supported on the native runtime; "
                                  "export from the local runtime (TELYS_RUNTIME=local)")

    def open_compact(self, path):
        raise NotImplementedError("compact serve artifacts are not yet supported on the native runtime; "
                                  "open with the local runtime (TELYS_RUNTIME=local)")

    # ── IVF-within-partition (PR-5; faiss clustering host-side, native exact rerank) ───────────────
    @_write
    def build_partition_ivf(self, index, min_rows, target_recall, max_nprobe=64):
        import faiss
        import numpy as np
        if _lib().ty_index_mutated(index._h):         # IVF builds over a clean base; fold deltas first
            self.compact(index)
        n = int(self._stats(index)["n"])
        if n == 0:
            return {}
        dim = index.dim
        base = np.empty(n * dim, np.float32); phys = np.empty(n, np.int64)
        _lib().ty_index_export_base(index._h, base.ctypes.data_as(ctypes.c_void_p),
                                    phys.ctypes.data_as(ctypes.c_void_p))
        base = base.reshape(n, dim)
        # de-intern each base row's partition key (native key_id -> value) to group rows per partition
        kids = np.empty(n, np.int64); _cv = np.empty(n * max(len(index.col_names), 1), np.int64)
        _lib().ty_index_export_rowmeta(index._h, np.ascontiguousarray(phys).ctypes.data_as(ctypes.c_void_p),
                                       int(n), kids.ctypes.data_as(ctypes.c_void_p),
                                       _cv.ctypes.data_as(ctypes.c_void_p))
        by_key: dict = {}
        for p in range(n):
            by_key.setdefault(index.key_i2v.get(int(kids[p])), []).append(p)
        index.ivf_keys, index.ivf_meta = set(), {}; _lib().ty_index_clear_ivf(index._h); built = {}
        for key, prows in by_key.items():
            if len(prows) < int(min_rows):
                continue
            pidx = np.asarray(prows, np.int64)
            V = np.ascontiguousarray(base[pidx], np.float32)
            lids = np.asarray([int(phys[p]) for p in prows], np.int64)
            nlist = max(1, int(round(np.sqrt(len(prows)))))
            km = faiss.Kmeans(dim, nlist, niter=10, seed=0, verbose=False)
            km.train(V)
            cent = np.ascontiguousarray(km.centroids.reshape(nlist, dim), np.float32)
            ci = faiss.IndexFlatIP(dim); ci.add(cent)
            _, asn = ci.search(V, 1); asn = asn[:, 0]
            crows = [[] for _ in range(nlist)]                  # cluster -> row indices into V
            clusters = [[] for _ in range(nlist)]              # cluster -> logical ids (for native rerank)
            for i in range(len(prows)):
                c = int(asn[i]); crows[c].append(i); clusters[c].append(int(lids[i]))
            nprobe = self._calibrate_nprobe(dim, cent, crows, V, lids, float(target_recall), int(max_nprobe))
            # store the IVF directory natively: flat cluster lids + nlist+1 offsets + centroids
            clus_flat = np.ascontiguousarray(np.asarray([lid for cl in clusters for lid in cl], np.int64))
            coff = np.zeros(nlist + 1, np.int32); acc = 0
            for c in range(nlist):
                coff[c] = acc; acc += len(clusters[c])
            coff[nlist] = acc; coff = np.ascontiguousarray(coff)
            cent_c = np.ascontiguousarray(cent, np.float32)
            _lib().ty_index_set_ivf(index._h, int(index.keymap[key]), int(nlist), int(nprobe),
                                    cent_c.ctypes.data_as(ctypes.c_void_p),
                                    clus_flat.ctypes.data_as(ctypes.c_void_p),
                                    coff.ctypes.data_as(ctypes.c_void_p), int(len(clus_flat)))
            index.ivf_keys.add(key); index.ivf_meta[key] = {"nlist": nlist, "nprobe": nprobe}
            built[key] = {"rows": len(prows), "nlist": nlist, "nprobe": nprobe}
        return built

    @staticmethod
    def _calibrate_nprobe(dim, cent, crows, V, lids, target, max_nprobe):
        """Smallest nprobe holding recall>=target. Queries are sampled from ACTUAL partition rows (+noise) —
        NOT centroids: centroid-centered queries sit on a cluster and are recalled at nprobe=1, which would
        wildly over-estimate recall and pick far too small an nprobe (matches LocalRuntime calibration)."""
        import faiss
        import numpy as np
        nlist = len(cent)
        rng = np.random.default_rng(0)
        m = min(128, len(V))
        a = rng.integers(0, len(V), size=m)
        Q = V[a] + 0.05 * rng.standard_normal((m, dim)).astype(np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
        Q = np.ascontiguousarray(Q, np.float32)
        fl = faiss.IndexFlatIP(dim); fl.add(V); _, truth = fl.search(Q, 10)
        cs = Q @ cent.T
        for nprobe in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
            if nprobe > nlist:
                break
            hit = 0.0
            for i in range(m):
                probe = np.argsort(cs[i])[::-1][:nprobe]
                cand = [r for c in probe for r in crows[int(c)]]
                if not cand:
                    continue
                cand = np.asarray(cand, np.int64)
                top = cand[np.argsort(V[cand] @ Q[i])[::-1][:10]]
                hit += len(set(top.tolist()) & set(truth[i].tolist())) / 10
            if hit / m >= target:
                return min(nprobe, max_nprobe)
        return min(max_nprobe, nlist)

    def _ivf_search(self, index, key_id, q, k):
        """Native IVF query: probe nprobe nearest centroids + gather + exact rerank — all in Mojo."""
        import numpy as np
        out_ids, out_sc = np.empty(max(k, 1), np.int64), np.empty(max(k, 1), np.float32)
        cnt = _lib().ty_index_search_ivf(
            index._h, int(key_id), q.ctypes.data_as(ctypes.c_void_p), k,
            out_ids.ctypes.data_as(ctypes.c_void_p), out_sc.ctypes.data_as(ctypes.c_void_p))
        v = out_ids[:cnt] >= 0
        return out_ids[:cnt][v].copy(), out_sc[:cnt][v].copy()

    @_read
    def partition_ivf_keys(self, index):
        return list(index.ivf_keys)      # @_read: ivf membership is reset under @_write (compact)

    @_write
    def set_partition_nprobe(self, index, key, nprobe):
        if key in index.ivf_keys:
            _lib().ty_index_set_ivf_nprobe(index._h, int(index.keymap[key]), int(nprobe))
            index.ivf_meta[key]["nprobe"] = int(nprobe)

    @_write
    def set_exact_crossover(self, index, rows):
        index.exact_crossover = int(rows)

    # ── provenance / enumeration (host-side metadata; MVCC-aware) ──────────────────────────────────
    @_read
    def columns_for(self, index, ids):
        import numpy as np
        ids = [int(x) for x in ids]
        nc = len(index.col_names)
        if not ids or nc == 0:
            return [{} for _ in ids]
        lids = np.ascontiguousarray(np.asarray(ids, np.int64))
        kids = np.empty(len(ids), np.int64)
        cv = np.empty(len(ids) * nc, np.int64)
        stride = int(_lib().ty_index_export_rowmeta(
            index._h, lids.ctypes.data_as(ctypes.c_void_p), int(len(ids)),
            kids.ctypes.data_as(ctypes.c_void_p), cv.ctypes.data_as(ctypes.c_void_p)))
        if stride != nc:                              # invariant: native ncols == host col_names (tripwire)
            raise RuntimeError(f"rowmeta stride {stride} != host ncols {nc}")
        cv = cv.reshape(len(ids), nc)
        return [{cname: index.col_i2v.get(cname, {}).get(int(cv[i, ci])) for ci, cname in enumerate(index.col_names)}
                for i in range(len(ids))]

    @_read
    def live_ids(self, index, where):
        import numpy as np
        col, val = _coerce_where(where)
        value = (val.item() if hasattr(val, "item") else val) if col is not None else None
        # resolve to (mode, col_idx, value_id); unknown key/value -> no matches
        if col is None:
            mode, ci, vid = 0, -1, 0
        elif col == index.key_name:
            mode, ci, vid = 1, -1, index.keymap.get(value)
        elif col in index.col_names:
            mode, ci, vid = 2, index.col_names.index(col), index.col_v2i.get(col, {}).get(value)
        else:
            raise NotImplementedError(f"NativeRuntime: unknown filter column {col!r}")
        if col is not None and vid is None:
            return []
        cap = int(_lib().ty_index_visible_count(index._h))
        if cap == 0:
            return []
        out = np.empty(cap, np.int64)
        cnt = int(_lib().ty_index_match_ids(index._h, int(mode), int(ci), int(vid),
                                            out.ctypes.data_as(ctypes.c_void_p)))
        return [int(x) for x in out[:cnt]]

    def kernel_info(self) -> dict:
        try:
            path = _resolve_native_lib()
            src = ("env TELYS_NATIVE_KERNEL" if os.environ.get("TELYS_NATIVE_KERNEL") or os.environ.get("AME_NATIVE_KERNEL")
                   else "installed" if _under_install_cache(path)
                   else "bundled" if (os.sep + "_runtime" + os.sep) in (path or "")
                   else "repo-relative/dev")
            return {"runtime": "native", "source": src, "path": path, "found": os.path.exists(path)}
        except Exception:  # noqa: BLE001
            return {"runtime": "native", "source": "native", "path": None, "found": False}

    def attach_embedder(self, index, embedder):
        # Embedding happens in the SDK facade (col.add_texts/search_text embed, then call build/insert/search
        # with vectors); the native engine only holds the reference. So text-backed ingest works on native too.
        index.embedder = embedder
