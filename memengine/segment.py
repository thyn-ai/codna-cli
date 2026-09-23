"""AME segment v0 (thin MVP) — vectors + row-ids as a Parquet file (pyarrow).

This is the minimal durable columnar segment used by the MVP: a `row_id` Int64 column + a
fixed-size-list<float32, D> `vector` column, written as Parquet (the build-on-Parquet decision,
internal/DECISIONS.md D-03 / internal/FORMAT-SPEC.md). The full segment format (superblock, sidecars,
WAL/manifest/MVCC) is specified in internal/FORMAT-SPEC.md and lands in Phase 1; this is the slice
needed to benchmark VectorCore-over-Parquet vs raw FAISS.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_segment(path: str, row_ids: np.ndarray, vectors: np.ndarray, compression: str = "zstd") -> None:
    n, d = vectors.shape
    flat = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1))
    vcol = pa.FixedSizeListArray.from_arrays(flat, d)
    table = pa.table({"row_id": pa.array(np.asarray(row_ids, dtype=np.int64)), "vector": vcol})
    pq.write_table(table, path, compression=compression)


def read_segment(path: str) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    row_ids = table.column("row_id").to_numpy(zero_copy_only=False).astype(np.int64)
    vcol = table.column("vector").combine_chunks()
    d = vcol.type.list_size
    flat = vcol.values.to_numpy(zero_copy_only=False).astype(np.float32)
    vectors = flat.reshape(len(row_ids), d)
    return row_ids, vectors
