"""LocalRuntime — the in-process implementation of the Telys runtime (telys-runtime, closed).

Implements the public `telys.runtime.RuntimeHandle` by wrapping `PartitionedVectorIndex`. This is the closed
side of the SDK↔engine seam (D-30): the public SDK loads it lazily via `telys.runtime.load_runtime()`. A
later sub-phase replaces this with a native runtime behind the same interface. `index` arguments are opaque
engine handles (today a `PartitionedVectorIndex`).
"""
from __future__ import annotations

from telys.runtime import FORMAT_VERSION as _SDK_FORMAT_VERSION
from telys.runtime import RuntimeHandle

from memengine.partitioned import PartitionedVectorIndex

# Fail closed if the engine's on-disk format diverges from the SDK's public format contract.
assert PartitionedVectorIndex.VERSION == _SDK_FORMAT_VERSION, (
    f"engine format v{PartitionedVectorIndex.VERSION} != SDK format v{_SDK_FORMAT_VERSION}")

FORMAT_VERSION = _SDK_FORMAT_VERSION


def _under_install_cache(path) -> bool:
    """True if `path` is inside the verified `telys runtime install` cache ($TELYS_HOME/runtime/...)."""
    import os
    try:
        from telys.paths import telys_home
        return bool(path) and os.path.abspath(path).startswith(os.path.join(telys_home(), "runtime") + os.sep)
    except Exception:  # noqa: BLE001
        return False


class LocalRuntime(RuntimeHandle):
    """In-process Python runtime: a thin pass-through to PartitionedVectorIndex."""

    FORMAT_VERSION = FORMAT_VERSION

    # lifecycle
    def new_index(self, dim, dtype): return PartitionedVectorIndex(dim, dtype)
    def open_index(self, path): return PartitionedVectorIndex.open(path)
    def save(self, index, path): return index.save(path)
    def sealed(self, index) -> bool: return index._sealed

    # ingest
    def attach_embedder(self, index, embedder): return index.attach_embedder(embedder)

    def build(self, index, vecs, iids, keys, key_name, columns, texts=None):
        return index.build(vecs, iids, keys, key_name=key_name, columns=columns, texts=texts)

    def insert(self, index, vecs, iids, keys, columns, texts=None):
        return index.insert(vecs, iids, keys, columns=columns, texts=texts)

    def update(self, index, iids, vecs, keys, columns, texts=None):
        return index.update(iids, vecs, keys=keys, columns=columns, texts=texts)

    # query
    def search(self, index, vector, top_k, where, explain, target_recall):
        return index.search(vector, top_k, where=where, explain=explain, target_recall=target_recall)

    def columns_for(self, index, ids): return index.columns_for(ids)
    def live_ids(self, index, where): return index.live_ids(where)

    # lexical (on-device BM25 code index)
    def build_lexical(self, index, k1=1.8, b=1.0): return index.build_lexical(k1=k1, b=b)

    def search_lexical(self, index, text, top_k, where, explain):
        return index.search_lexical(text, top_k, where=where, explain=explain)

    # maintain
    def delete(self, index, iids): return index.delete(iids)
    def compact(self, index): return index.compact()
    def snapshot(self, index): return index.snapshot()
    def stats(self, index) -> dict: return index.stats()

    # compact serve artifacts — the engine already seals these; LocalRuntime passes straight through
    def export_compact(self, index, path, mode): return index.save(path, compact=mode)
    def open_compact(self, path): return PartitionedVectorIndex.open(path)

    # physical tuning state
    def build_partition_ivf(self, index, min_rows, target_recall, max_nprobe=64):
        return index.build_partition_ivf(min_rows=min_rows, target_recall=target_recall, max_nprobe=max_nprobe)

    def set_exact_crossover(self, index, rows): index.exact_crossover = int(rows)
    def partition_ivf_keys(self, index): return list(index.pivf.keys())
    def set_partition_nprobe(self, index, key, nprobe): index.pivf[key]["nprobe"] = int(nprobe)

    # default zero-config tuner
    def default_tuner(self):
        from memengine.tuning_heuristic import HeuristicTuner
        return HeuristicTuner()

    # diagnostics — kernel resolution, so the SDK/CLI need not import engine internals
    def kernel_info(self) -> dict:
        import os
        from memengine import mojo_backend as mb
        path = mb._LIB_PATH
        src = ("env TELYS_KERNEL" if os.environ.get("TELYS_KERNEL")
               else "env AME_KERNEL" if os.environ.get("AME_KERNEL")
               else "installed" if _under_install_cache(path)
               else "bundled" if (os.sep + "_runtime" + os.sep) in (path or "")
               else "repo-relative/dev")
        return {"runtime": "local", "source": src, "path": path, "found": bool(path and os.path.exists(path))}
