"""Defense-in-depth input validation at the Python↔Mojo FFI boundary.

The native runtime is in-process: a bad pointer/length/count that crosses into Mojo is an UNCATCHABLE
SIGSEGV/SIGABRT that takes down the whole process (and, in the self-host server, every connected client).
So every value is validated HERE, before any ctypes call, turning malformed input into a clean Python
exception. The Mojo kernel keeps its own clamps as a redundant second layer for direct callers.

Raise vs clamp follows the established split: dimension/finiteness/overflow are programmer/data errors → raise;
top-k is a request knob → clamp (mirrors server._topk at telys/server.py).
"""
from __future__ import annotations

import numpy as np

INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)


def fits_i32(n: int, what: str = "count") -> int:
    """Ensure a length/count fits a C int32 arg (a >2^31 value silently wraps negative → Mojo loop UB)."""
    n = int(n)
    if n < INT32_MIN or n > INT32_MAX:
        raise OverflowError(f"{what}={n} does not fit int32 [{INT32_MIN}, {INT32_MAX}]")
    return n


def require_handle(h, what: str = "index") -> None:
    """Reject a NULL/freed native handle before it is dereferenced in Mojo."""
    if not h:
        raise RuntimeError(f"{what} handle is null/closed — the native index is not open")


def require_finite(arr: np.ndarray, what: str = "vectors") -> np.ndarray:
    """Reject NaN/Inf: they propagate through the dot product and silently corrupt top-k ordering
    (NaN comparisons are always false), which is a correctness hole, not a crash."""
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} contain NaN or Inf — only finite float32 values are allowed")
    return arr


def require_matrix(v: np.ndarray, dim: int, what: str = "vectors") -> np.ndarray:
    """Enforce the (n, dim) contract: the kernel strides buffers by `dim`, so a wrong width OOB-reads."""
    if v.ndim != 2 or v.shape[1] != dim:
        raise ValueError(f"{what} must be (n, {dim}); got shape {tuple(v.shape)}")
    return v


def clamp_topk(k, cap: int) -> int:
    """Clamp a requested top-k to [0, cap] so the caller-sized output buffer can never be under-allocated
    relative to what the kernel writes."""
    return max(0, min(int(k), int(cap)))


def check_count(cnt: int, cap: int, what: str = "result count") -> int:
    """Tripwire on the value the kernel RETURNS: a cnt outside [0, cap] means the kernel wrote past the
    caller's buffer (a kernel bug) — fail loud instead of slicing garbage."""
    cnt = int(cnt)
    if cnt < 0 or cnt > int(cap):
        raise RuntimeError(f"native {what}={cnt} outside [0, {cap}] — output buffer contract violated")
    return cnt
