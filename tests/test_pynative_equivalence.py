"""Equivalence of the pure-NumPy reference against the native Mojo kernel.

These tests are the evidence for the central claim of the Windows work: the kernel is an ACCELERATOR, so a
platform without it runs the same retrieval, not a different one. They are written to be meaningful in both
directions:

* WITH the kernel available (macOS arm64 / manylinux x86_64) every primitive is compared function-for-function
  against the real shared library, which is the only way to prove the reference is a reference and not a guess.
* WITHOUT it (any other platform, or this machine with ``TELYS_KERNEL`` pointed at nothing) the reference is
  compared against the crate of behaviours the kernel is documented to have, so the suite still fails if
  someone changes the arithmetic.

The comparisons are deliberately two-tiered, matching what is actually guaranteed:

* INTEGER-domain work (n-gram hashing, int8 quantisation, token hashing) must be BIT-IDENTICAL. Anything less
  is a bug.
* FLOAT-domain work (inner products, BM25 accumulation) must agree to float32 rounding and produce the SAME
  RANKING. The kernel sums along SIMD lanes and NumPy sums pairwise, so demanding bit equality would assert
  something neither implementation can promise.

Skip conditions are explicit, so a run on a machine without the kernel reports "skipped (no kernel)" rather
than silently passing on zero assertions.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ── load the module under test BY PATH, with no global effect ──────────────────────────────────────
#
# `_pynative` is the thing under test, so it must be the VENDORED copy — but this module must not touch `sys.path`
# or `sys.modules` while it is being imported: pytest imports every test module during collection, and a
# module-level pin leaks into the whole session and silently changes what unrelated tests import. (That mistake
# briefly made eight unrelated memory tests "pass" by accident, which is exactly the kind of false signal this work
# has to avoid.)
#
# It also cannot rely on `import memengine._pynative`: collection order is not ours, and any test that imports
# `telys` pushes the verified runtime's `pysite` — which ships a `memengine` WITHOUT `_pynative` — to the front of
# sys.path. Loading by file path is local, deterministic, and cannot resolve to the wrong engine.
#
# Safe to load standalone: `_pynative` imports only `math` and `numpy` (no intra-package imports), which is itself a
# property this suite asserts (`test_reference_has_no_native_dependency`).
_CLI_DIR = Path(__file__).resolve().parents[1]
_VENDORED_MEMENGINE = (_CLI_DIR / "memengine").resolve()
_PYNATIVE_PATH = _VENDORED_MEMENGINE / "_pynative.py"



VP = ctypes.c_void_p
I32 = ctypes.c_int32
F32 = ctypes.c_float

# Float32 tolerance for a dimension-d dot product: every term is O(1), so the accumulated error grows with d.
SCORE_RTOL = 1e-5
SCORE_ATOL = 1e-5


_SEEDED_NAMES = ("memengine", "memengine._pynative", "memengine._pykernel", "memengine.mojo_backend",
                 "telys", "telys.embedding")


def _sdk_embedding_importable() -> bool:
    """True when ``from telys.embedding import ...`` would work RIGHT NOW.

    A full-suite session can leave either binding unusable: ``sys.modules["telys"] = None`` (a test that wants
    "the SDK is missing" leaves the sentinel behind when its monkeypatch is undone early) or a bare
    ``SimpleNamespace`` standing in for the package (which has no ``__path__``, so submodules stop resolving).
    Both make the import below raise, which is the 19-test embedder cluster. Detect, do not assume.
    """
    try:
        from telys.embedding import EmbeddingProvider  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable as-is"
        return False


def _sdk_embedding_path() -> Path | None:
    """The installed SDK's ``telys/embedding.py``, found WITHOUT trusting ``sys.modules`` or the front of sys.path.

    ``sys.path`` alone is not enough: a test may have replaced it wholesale, and the poisoning this guards against
    is exactly the state where the ordinary import machinery cannot answer. So this scans the live path plus the
    interpreter's own site directories, and finally the location of a real ``telys`` already imported.
    """
    roots: list[str] = list(sys.path)
    try:
        import site

        roots.extend(site.getsitepackages())
        roots.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001 - site is present everywhere we support; degrade to sys.path only
        pass
    imported = sys.modules.get("telys")
    if imported is not None and getattr(imported, "__file__", None):
        roots.append(str(Path(imported.__file__).parent.parent))

    for root in roots:
        if not root:
            continue
        candidate = Path(root) / "telys" / "embedding.py"
        if candidate.is_file():
            return candidate
    return None


def _load_by_path_registered(path: Path, module_name: str):
    """`_load_by_path`, but the module is in ``sys.modules`` BEFORE its body runs.

    ``telys/embedding.py`` uses ``@dataclass``, and dataclasses look their own class's module up in
    ``sys.modules`` while processing the class. Executing it unregistered leaves a None lookup and dies with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'`` — so registration is part of loading it.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _seed_sdk_modules(saved: dict) -> None:
    """Make ``from telys.embedding import ...`` resolvable for the duration of one vendored load.

    ``memengine/embedding.py`` imports the embedding INTERFACES from ``telys.embedding`` at import time, so a
    poisoned binding there breaks the comparison even though everything under test is vendored. This re-binds
    ``telys`` / ``telys.embedding`` to the REAL installed files by path, records what was there, and leaves the
    caller to restore it (see `_vendored_module`'s unconditional ``finally``).

    It does nothing when the ordinary import already works, so a healthy session is untouched.
    """
    if _sdk_embedding_importable():
        return
    path = _sdk_embedding_path()
    if path is None:  # pragma: no cover - an install without the SDK cannot reach these tests at all
        return
    import types

    for name in ("telys", "telys.embedding"):
        saved.setdefault(name, sys.modules.get(name, _MISSING))

    package = types.ModuleType("telys")
    package.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
    sys.modules["telys"] = package
    sys.modules.pop("telys.embedding", None)
    sys.modules["telys.embedding"] = _load_by_path_registered(path, "telys.embedding")


_MISSING = object()


def _vendored_module(filename: str, module_name: str, *, seed_package: bool = False):
    """Load a module from the VENDORED `memengine` directory BY PATH.

    Why by path: a full-suite run may already have resolved `memengine` to the verified runtime's `pysite` (which
    ships NO `_pynative`), and Python caches that resolution — so `import memengine.lexical` could fail or hand back
    a different engine depending on collection order. Loading by path is local and deterministic.

    `seed_package=True` additionally exposes the vendored package under the real `memengine` name WHILE this module
    loads and restores the previous bindings afterwards. Needed for modules that do intra-package imports at import
    time (`lexical.py` lazily imports `memengine.mojo_backend._lib`; `mojo_backend.py` imports `memengine._pykernel`).
    It also re-binds `telys`/`telys.embedding` by path when a previous test left them unusable, because
    `embedding.py` imports its interfaces from there. The restore is unconditional: leaving `memengine.*` or
    `telys.*` rewired would change what every later test imports.
    """
    path = _VENDORED_MEMENGINE / filename
    if not path.is_file():  # pragma: no cover
        pytest.skip(f"vendored module missing: {path}")

    saved: dict = {}
    if seed_package:
        import types

        saved = {name: sys.modules.get(name, _MISSING) for name in _SEEDED_NAMES}
        _seed_sdk_modules(saved)
        package = types.ModuleType("memengine")
        package.__path__ = [str(_VENDORED_MEMENGINE)]  # type: ignore[attr-defined]
        sys.modules["memengine"] = package
        sys.modules["memengine._pynative"] = _load_by_path(_VENDORED_MEMENGINE / "_pynative.py",
                                                          "memengine._pynative")
        sys.modules["memengine._pykernel"] = _load_by_path(_VENDORED_MEMENGINE / "_pykernel.py",
                                                           "memengine._pykernel")
        sys.modules["memengine.mojo_backend"] = _load_by_path(_VENDORED_MEMENGINE / "mojo_backend.py",
                                                              "memengine.mojo_backend")
    try:
        return _load_by_path(path, module_name)
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_by_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"cannot load {path}", allow_module_level=True)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PN = _load_by_path(_PYNATIVE_PATH, "memengine_pynative_under_test")

VP = ctypes.c_void_p
I32 = ctypes.c_int32
F32 = ctypes.c_float

# Float32 tolerance for a dimension-d dot product: every term is O(1), so the accumulated error grows with d.
SCORE_RTOL = 1e-5
SCORE_ATOL = 1e-5


def _native_lib():
    """The real shared library, or None. Never the PyKernel stand-in -- that would be comparing it to itself.

    Resolved from the VENDORED `mojo_backend` (loaded by path, so it does not depend on what `memengine` already means
    in this process) and never at module import: collection must not import `memengine` at all.
    """
    try:
        mb = _vendored_module("mojo_backend.py", "memengine_mojo_backend_under_test", seed_package=True)
        if not os.path.exists(mb._LIB_PATH):  # noqa: SLF001 - the module's own resolution result
            return None
        if mb.backend_kind() != "native":
            return None
        return mb._lib()
    except (OSError, ImportError, AttributeError):  # pragma: no cover - broken install / absent module
        return None


# `skipif` needs a static-ish condition, so evaluate it from a cached helper that is safe to call during
# collection: it catches every failure and reports "not available", which makes the whole module skip rather than
# error. The rich comparison then happens inside the tests, where imports are properly scoped.
def _native_available() -> bool:
    try:
        if not Path(PN.__file__).resolve().is_relative_to(_VENDORED_MEMENGINE):
            return False
        return _native_lib() is not None
    except Exception:  # noqa: BLE001 - an unusable environment means "no native comparison possible"
        return False


native = pytest.mark.skipif(not _native_available(), reason="native Mojo kernel not available on this platform")


def _assert_same_topk(got_ids, got_scores, want_ids, want_scores, *, k: int, label: str) -> None:
    """Identical ranking and float32-equal scores. ``-1`` padding is trimmed from both sides first."""
    gi = np.asarray(got_ids).reshape(-1)
    wi = np.asarray(want_ids).reshape(-1)
    gi = gi[gi >= 0]
    wi = wi[wi >= 0]
    assert len(gi) == len(wi), f"{label}: hit count differs ({len(gi)} vs {len(wi)})"
    assert np.array_equal(gi, wi), f"{label}: ranking differs\n  reference={gi[:k]}\n  kernel   ={wi[:k]}"
    if len(gi):
        np.testing.assert_allclose(
            np.asarray(got_scores).reshape(-1)[: len(gi)],
            np.asarray(want_scores).reshape(-1)[: len(gi)],
            rtol=SCORE_RTOL,
            atol=SCORE_ATOL,
            err_msg=f"{label}: scores differ",
        )


# ── flat top-k ────────────────────────────────────────────────────────────────────────────────
@native
def test_flat_topk_matches_the_kernel() -> None:
    lib = _native_lib()
    rng = np.random.default_rng(11)
    n, d, k = 997, 24, 10
    base = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)

    out_ids = np.full(k, -1, np.int32)
    out_dist = np.zeros(k, np.float32)
    lib.ame_flat_topk(
        base.ctypes.data_as(VP), n, d, query.ctypes.data_as(VP), k,
        out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
    )
    ref_ids, ref_scores = PN.flat_topk(base, n, d, query, k)
    _assert_same_topk(ref_ids, ref_scores, out_ids, out_dist, k=k, label="flat_topk")


@native
def test_flat_topk_i8_matches_the_kernel() -> None:
    lib = _native_lib()
    rng = np.random.default_rng(12)
    n, d, k = 500, 32, 8
    base = rng.standard_normal((n, d)).astype(np.float32)
    base_q8 = np.empty(n * d, np.int8)
    lib.ame_quantize_i8(base.ctypes.data_as(VP), n, d, base_q8.ctypes.data_as(VP))
    query = rng.standard_normal(d).astype(np.float32)

    out_ids = np.full(k, -1, np.int32)
    out_dist = np.zeros(k, np.float32)
    qq8 = np.empty(d, np.int8)
    lib.ame_flat_topk_i8(
        base_q8.ctypes.data_as(VP), n, d, query.ctypes.data_as(VP), qq8.ctypes.data_as(VP), k,
        out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
    )
    ref_ids, ref_scores = PN.flat_topk_i8(base_q8, n, d, query, k)
    _assert_same_topk(ref_ids, ref_scores, out_ids, out_dist, k=k, label="flat_topk_i8")


@native
@pytest.mark.parametrize("kind", ["f16", "q8"])
def test_flat_topk_quantised_rerank_matches_the_kernel(kind: str) -> None:
    lib = _native_lib()
    rng = np.random.default_rng(13)
    n, d, k, cap = 600, 24, 8, 32
    base = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)
    out_ids = np.full(k, -1, np.int32)
    out_dist = np.zeros(k, np.float32)
    scratch_id = np.empty(cap, np.int32)

    if kind == "f16":
        shadow = np.empty(n * d, np.float16)
        lib.ame_quantize_f16(base.ctypes.data_as(VP), n, d, shadow.ctypes.data_as(VP))
        scratch_s = np.empty(cap, np.float32)
        lib.ame_flat_topk_f16(
            base.ctypes.data_as(VP), shadow.ctypes.data_as(VP), n, d, query.ctypes.data_as(VP), k, cap,
            scratch_id.ctypes.data_as(VP), scratch_s.ctypes.data_as(VP),
            out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
        )
        ref_ids, ref_scores = PN.flat_topk_f16(base, shadow, n, d, query, k, cap)
    else:
        shadow = np.empty(n * d, np.int8)
        lib.ame_quantize_i8(base.ctypes.data_as(VP), n, d, shadow.ctypes.data_as(VP))
        scratch_s = np.empty(cap, np.int32)
        qq8 = np.empty(d, np.int8)
        lib.ame_flat_topk_q8(
            base.ctypes.data_as(VP), shadow.ctypes.data_as(VP), n, d, query.ctypes.data_as(VP),
            qq8.ctypes.data_as(VP), k, cap, scratch_id.ctypes.data_as(VP), scratch_s.ctypes.data_as(VP),
            out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
        )
        ref_ids, ref_scores = PN.flat_topk_q8(base, shadow, n, d, query, k, cap)

    _assert_same_topk(ref_ids, ref_scores, out_ids, out_dist, k=k, label=f"flat_topk_{kind}")


# ── quantisers (integer domain: bit-identical) ────────────────────────────────────────────────
@native
def test_quantize_i8_is_bit_identical() -> None:
    lib = _native_lib()
    rng = np.random.default_rng(14)
    n, d = 200, 16
    # Include exact .5 boundaries to exercise round-half-away-from-zero.
    src = np.concatenate([
        rng.standard_normal((n - 2, d)),
        np.full((1, d), 0.5 / 127.0),
        np.full((1, d), -0.5 / 127.0),
    ]).astype(np.float32)

    kernel_q8 = np.empty(n * d, np.int8)
    lib.ame_quantize_i8(src.ctypes.data_as(VP), n, d, kernel_q8.ctypes.data_as(VP))
    ref_q8 = np.empty(n * d, np.int8)
    PN.quantize_i8(src, n, d, ref_q8)
    assert np.array_equal(kernel_q8, ref_q8), "int8 quantisation must be bit-identical (round-half-away-from-zero)"


@native
def test_quantize_f16_is_bit_identical() -> None:
    lib = _native_lib()
    rng = np.random.default_rng(15)
    n, d = 200, 16
    src = rng.standard_normal((n, d)).astype(np.float32)
    kernel_f16 = np.empty(n * d, np.float16)
    lib.ame_quantize_f16(src.ctypes.data_as(VP), n, d, kernel_f16.ctypes.data_as(VP))
    ref_f16 = np.empty(n * d, np.float16)
    PN.quantize_f16(src, n, d, ref_f16)
    assert np.array_equal(kernel_f16, ref_f16), "float16 narrowing must be bit-identical (IEEE-754)" 


# ── IVF ───────────────────────────────────────────────────────────────────────────────────────
def _ivf_fixture(seed: int, n: int, d: int, nlist: int):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d)).astype(np.float32)
    assign = rng.integers(0, nlist, n)
    order = np.argsort(assign, kind="stable")
    base_sorted = np.ascontiguousarray(base[order])
    counts = np.bincount(assign, minlength=nlist)
    offsets = np.zeros(nlist + 1, np.int32)
    offsets[1:] = np.cumsum(counts)
    centroids = rng.standard_normal((nlist, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)
    return base_sorted, offsets, centroids, query


@native
@pytest.mark.parametrize("variant", ["f32", "f16", "q8"])
def test_ivf_search_matches_the_kernel(variant: str) -> None:
    lib = _native_lib()
    n, d, nlist, nprobe, k, cap = 800, 16, 32, 6, 10, 40
    base_sorted, offsets, centroids, query = _ivf_fixture(16, n, d, nlist)
    out_ids = np.full(k, -1, np.int32)
    out_dist = np.zeros(k, np.float32)
    scratch_id = np.empty(cap, np.int32)

    if variant == "f32":
        lib.ame_ivf_search(
            centroids.ctypes.data_as(VP), nlist, base_sorted.ctypes.data_as(VP),
            offsets.ctypes.data_as(VP), d, query.ctypes.data_as(VP), nprobe, k,
            out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
        )
        ref_ids, ref_scores = PN.ivf_search(centroids, nlist, base_sorted, offsets, d, query, nprobe, k)
    elif variant == "f16":
        shadow = np.empty(n * d, np.float16)
        lib.ame_quantize_f16(base_sorted.ctypes.data_as(VP), n, d, shadow.ctypes.data_as(VP))
        scratch_s = np.empty(cap, np.float32)
        lib.ame_ivf_search_f16(
            centroids.ctypes.data_as(VP), nlist, base_sorted.ctypes.data_as(VP), shadow.ctypes.data_as(VP),
            offsets.ctypes.data_as(VP), d, query.ctypes.data_as(VP), nprobe, k, cap,
            scratch_id.ctypes.data_as(VP), scratch_s.ctypes.data_as(VP),
            out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
        )
        ref_ids, ref_scores = PN.ivf_search_f16(
            centroids, nlist, base_sorted, shadow, offsets, d, query, nprobe, k, cap
        )
    else:
        shadow = np.empty(n * d, np.int8)
        lib.ame_quantize_i8(base_sorted.ctypes.data_as(VP), n, d, shadow.ctypes.data_as(VP))
        scratch_s = np.empty(cap, np.int32)
        qq8 = np.empty(d, np.int8)
        lib.ame_ivf_search_q8(
            centroids.ctypes.data_as(VP), nlist, base_sorted.ctypes.data_as(VP), shadow.ctypes.data_as(VP),
            offsets.ctypes.data_as(VP), d, query.ctypes.data_as(VP), nprobe, k, cap,
            qq8.ctypes.data_as(VP), scratch_id.ctypes.data_as(VP), scratch_s.ctypes.data_as(VP),
            out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
        )
        ref_ids, ref_scores = PN.ivf_search_q8(
            centroids, nlist, base_sorted, shadow, offsets, d, query, nprobe, k, cap
        )

    _assert_same_topk(ref_ids, ref_scores, out_ids, out_dist, k=k, label=f"ivf_{variant}")


@native
def test_kmeans_build_matches_the_kernel() -> None:
    """Same centroids and same labels. This is a BUILD primitive: a divergence would change every IVF result."""
    lib = _native_lib()
    rng = np.random.default_rng(17)
    for n, d, k, iters in ((200, 8, 10, 10), (64, 16, 8, 5), (300, 24, 12, 8)):
        data = rng.standard_normal((n, d)).astype(np.float32)
        kernel_c = np.zeros((k, d), np.float32)
        kernel_l = np.zeros(n, np.int32)
        lib.ame_kmeans_build(
            data.ctypes.data_as(VP), n, d, k, iters,
            kernel_c.ctypes.data_as(VP), kernel_l.ctypes.data_as(VP),
        )
        ref_c, ref_l = PN.kmeans_build(data, n, d, k, iters)
        np.testing.assert_allclose(ref_c, kernel_c, rtol=SCORE_RTOL, atol=SCORE_ATOL,
                                   err_msg=f"kmeans centroids differ (n={n}, k={k})")
        assert np.array_equal(ref_l, kernel_l), f"kmeans labels differ (n={n}, k={k})"


@native
def test_flat_topk_pq_matches_the_kernel() -> None:
    lib = _native_lib()
    rng = np.random.default_rng(21)
    n, m, ksub, dsub, k = 120, 4, 16, 8, 10
    d = m * dsub
    codes = rng.integers(0, ksub, size=n * m).astype(np.uint8)
    codebook = rng.standard_normal(m * ksub * dsub).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)
    out_ids = np.full(k, -1, np.int32)
    out_dist = np.zeros(k, np.float32)
    lut = np.zeros(m * ksub, np.float32)
    lib.ame_flat_topk_pq(
        codes.ctypes.data_as(VP), n, m, ksub, dsub, codebook.ctypes.data_as(VP),
        query.ctypes.data_as(VP), lut.ctypes.data_as(VP), k,
        out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
    )
    ref_ids, ref_scores = PN.flat_topk_pq(codes, n, m, ksub, dsub, codebook, query, k)
    _assert_same_topk(ref_ids, ref_scores, out_ids, out_dist, k=k, label="flat_topk_pq")


# ── embedders (integer hashing: bit-identical) ────────────────────────────────────────────────
def _vendored_embedder(name: str):
    """An embedder class from the VENDORED engine, loaded deterministically.

    These tests compare against the KERNEL-backed embedder, so it must be the vendored one. A plain
    `from memengine.embedding import ...` is not good enough in a full-suite run: another test may already have
    pushed the verified runtime's `pysite` onto sys.path, and that `memengine` is built without `_pynative` — so the
    import could resolve elsewhere or fail outright depending on collection order.

    `embedding.py` does `from memengine import _validate` and `from memengine.mojo_backend import _lib` at import
    time, hence `seed_package=True`.
    """
    module = _vendored_module("embedding.py", f"memengine_embedding_under_test_{name}", seed_package=True)
    return getattr(module, name)


@pytest.mark.parametrize(
    "text",
    ["refund ledger", "hello world", "", "a", "ab", "abc", "abcd", "A_B-C.D xyz",
     "def lost_cookie():\n    return 'cookiejar'\n", "x" * 200],
)
def test_bigram_embed_is_bit_identical(text: str) -> None:
    """Bigram hashing is integer arithmetic, so the reference must reproduce the kernel EXACTLY."""
    if not _native_available():
        # No kernel: assert the reference against the documented contract (buckets, L2, zero vector).
        vec = PN.bigram_embed(text.encode())
        assert vec.shape == (PN.bigram_dim(),)
        # A bigram embedding needs at least two bytes; shorter input has no bigram to hash, so the all-zero
        # vector IS the documented answer (it sorts last under inner product, never a normalization error).
        if len(text.encode()) < 2:
            assert not vec.any()
        else:
            assert vec.any() and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6
        return
    embedder = _vendored_embedder("AlgentaBigramEmbedder")()
    np.testing.assert_array_equal(embedder._one(text), PN.bigram_embed(text.encode()))  # noqa: SLF001


@pytest.mark.parametrize(
    "text",
    ["refund ledger", "hello world", "", "a", "ab", "abc", "A_B-C.D xyz",
     "def lost_cookie():\n    return 'cookiejar'\n", "x" * 200],
)
def test_multigram_embed_is_bit_identical(text: str) -> None:
    if not _native_available():
        vec = PN.multigram_embed(text.encode())
        assert vec.shape == (PN.multigram_dim(),)
        assert PN.multigram_dim() == PN.UNIGRAM_DIM + PN.BIGRAM_DIM + PN.TRIGRAM_DIM
        return
    embedder = _vendored_embedder("AlgentaMultigramEmbedder")()
    np.testing.assert_array_equal(embedder._one(text), PN.multigram_embed(text.encode()))  # noqa: SLF001


# ── code-lexical tokenizer + BM25 ─────────────────────────────────────────────────────────────
TOKENIZER_CASES = [
    "def lost_cookie(): return cookiejar",
    "class RefundService: process_ledger(x)",
    "migrate_ledger_to_v2 schema",
    "A_B_C d.e-f 123abc",
    "HTTPServer XMLHttpRequest",
    "reconciles entries after migration",
    "",
    "x" * 50,
    "foo_sses bar_ches",
    "ALLCAPS mixedCase",
    "def _private__name(self, order_refunded):",
]


def _vendored_lexical():
    """The VENDORED `memengine.lexical` module, with its intra-package imports seeded (see `_vendored_module`)."""
    return _vendored_module("lexical.py", "memengine_lexical_under_test", seed_package=True)


@pytest.mark.parametrize("text", TOKENIZER_CASES)
def test_code_tokenize_matches_the_kernel(text: str) -> None:
    """Token ORDER is part of the contract: it determines postings order and therefore tie-breaking."""
    lexical = _vendored_lexical()
    stop = {PN.fnv1a_signed(word) for word in lexical.STOP}
    reference = PN.code_tokenize_hashes(text, stop)
    if not _native_available():
        assert isinstance(reference, list)
        assert all(isinstance(h, int) for h in reference)
        return
    assert lexical.tokenize_hashes(text) == reference


def _bm25_corpus(seed: int, n_docs: int = 60):
    rng = np.random.default_rng(seed)
    docs = [[int(x) for x in rng.integers(0, 40, size=int(rng.integers(3, 14)))] for _ in range(n_docs)]
    lex = _vendored_lexical().LexicalCodeIndex().build_from_hashes([str(i) for i in range(n_docs)], docs)
    return lex


def test_bm25_csr_topk_matches_the_kernel_path() -> None:
    """The postings query path must agree with the pure-python postings scan, kernel or not.

    Runs on EVERY platform: it compares the index's own ``search`` (kernel when present, reference otherwise) with
    ``_search_postings`` (always pure NumPy). That makes the fallback exercised on macOS too, rather than only on a
    machine that happens to lack a kernel.
    """
    lex = _bm25_corpus(2)
    q = "refund ledger migration"
    got = lex.search(q, 10)
    ref = lex._search_postings(q, 10)  # noqa: SLF001 - the reference this test exists to pin
    assert [i for i, _ in got] == [i for i, _ in ref], "kernel and postings paths disagree on ranking"
    for (_, a), (_, b) in zip(got, ref):
        assert abs(a - b) <= SCORE_ATOL + SCORE_RTOL * abs(b)


@native
def test_bm25_kernels_match_the_module_reference() -> None:
    lib = _native_lib()
    lex = _bm25_corpus(3)
    query_hashes = [1, 5, 7, 12]
    k = 10

    starts, ends, idfs = [], [], []
    for h in sorted(set(query_hashes)):
        rng_ = lex._term_range.get(h)  # noqa: SLF001
        if rng_ is None:
            continue
        st, cnt = rng_
        starts.append(st)
        ends.append(st + cnt)
        idfs.append(lex._bm.idf[h])  # noqa: SLF001

    qs = np.asarray(starts, np.int32)
    qe = np.asarray(ends, np.int32)
    qi = np.asarray(idfs, np.float32)
    out_ids = np.full(k, -1, np.int32)
    out_scores = np.zeros(k, np.float32)
    scratch = np.empty(len(lex.ids), np.float32)
    lib.ame_bm25_csr_topk(
        lex._post_docs.ctypes.data_as(VP), lex._post_tfs.ctypes.data_as(VP),  # noqa: SLF001
        qs.ctypes.data_as(VP), qe.ctypes.data_as(VP), qi.ctypes.data_as(VP), len(starts),
        lex._doc_denom.ctypes.data_as(VP), len(lex.ids), scratch.ctypes.data_as(VP),  # noqa: SLF001
        F32(lex.k1), k, out_ids.ctypes.data_as(VP), out_scores.ctypes.data_as(VP),
    )
    ref_ids, ref_scores = PN.bm25_csr_topk(
        lex._post_docs, lex._post_tfs, qs, qe, qi, len(starts),  # noqa: SLF001
        lex._doc_denom, len(lex.ids), lex.k1, k,  # noqa: SLF001
    )
    _assert_same_topk(ref_ids, ref_scores, out_ids, out_scores, k=k, label="bm25_csr_topk")

    # And the per-document scan kernel.
    uq = sorted(set(query_hashes))
    qa = np.asarray(uq, np.int64)
    qidf = np.asarray([lex._bm.idf.get(h, 0.0) for h in uq], np.float32)  # noqa: SLF001
    out_ids2 = np.full(k, -1, np.int32)
    out_scores2 = np.zeros(k, np.float32)
    lib.ame_bm25_topk(
        lex._packed.ctypes.data_as(VP), lex._offs.ctypes.data_as(VP), len(lex.ids),  # noqa: SLF001
        qa.ctypes.data_as(VP), qidf.ctypes.data_as(VP), len(uq),
        F32(lex.k1), F32(lex.b), F32(lex._bm.avgdl), k,  # noqa: SLF001
        out_ids2.ctypes.data_as(VP), out_scores2.ctypes.data_as(VP),
    )
    ref_ids2, ref_scores2 = PN.bm25_topk(
        lex._packed, lex._offs, len(lex.ids), qa, qidf, len(uq),  # noqa: SLF001
        lex.k1, lex.b, lex._bm.avgdl, k,  # noqa: SLF001
    )
    _assert_same_topk(ref_ids2, ref_scores2, out_ids2, out_scores2, k=k, label="bm25_topk")


# ── NSW ───────────────────────────────────────────────────────────────────────────────────────
@native
def test_nsw_build_and_search_match_the_kernel() -> None:
    """Graph build is deterministic given the same inputs, so the neighbour table must be identical."""
    lib = _native_lib()
    rng = np.random.default_rng(31)
    n, d, m, efc, ef, k = 200, 12, 16, 40, 30, 10
    base = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)

    ref_neighbors, ref_degree = PN.nsw_build(base, n, d, m, efc)

    kernel_neighbors = np.full(n * m, -1, np.int32)
    kernel_degree = np.zeros(n, np.int32)
    visited = np.zeros(n, np.int32)
    # The build's beam is sized by `ef_construction`, not `ef_search`: the kernel writes up to `efc` result
    # slots while inserting. Allocating `ef` here is a buffer overflow (and a bus error on teardown).
    res_id = np.empty(max(ef, efc), np.int32)
    res_ip = np.empty(max(ef, efc), np.float32)
    lib.ame_nsw_build(
        base.ctypes.data_as(VP), n, d, m, efc, kernel_neighbors.ctypes.data_as(VP),
        kernel_degree.ctypes.data_as(VP), visited.ctypes.data_as(VP),
        res_id.ctypes.data_as(VP), res_ip.ctypes.data_as(VP),
    )
    assert np.array_equal(ref_neighbors, kernel_neighbors.reshape(n, m)), "NSW neighbour tables differ"
    assert np.array_equal(ref_degree, kernel_degree), "NSW degrees differ"

    out_ids = np.full(k, -1, np.int32)
    out_dist = np.zeros(k, np.float32)
    visited_search = np.zeros(n, np.int32)
    neighbors_flat = np.ascontiguousarray(ref_neighbors.reshape(-1), np.int32)
    degree_flat = np.ascontiguousarray(ref_degree, np.int32)
    lib.ame_nsw_search(
        base.ctypes.data_as(VP), n, d, m, neighbors_flat.ctypes.data_as(VP), degree_flat.ctypes.data_as(VP),
        0, query.ctypes.data_as(VP), ef, k, visited_search.ctypes.data_as(VP), 1,
        res_id.ctypes.data_as(VP), res_ip.ctypes.data_as(VP),
        out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP),
    )
    ref_ids, ref_scores = PN.nsw_search(base, n, d, m, ref_neighbors, ref_degree, 0, query, ef, k)
    _assert_same_topk(ref_ids, ref_scores, out_ids, out_dist, k=k, label="nsw_search")


# ── platform-independent sanity (no kernel required) ─────────────────────────────────────────
def test_reference_has_no_native_dependency() -> None:
    """The reference must be importable and usable with the kernel entirely unavailable.

    This is the load-bearing property for Windows: if :mod:`memengine._pynative` ever grows an import that only
    the kernel build provides, this test fails on a machine that HAS the kernel too, which is the only reason it
    catches the regression before a Windows run does.
    """
    assert not hasattr(PN, "_lib"), "the reference module must not import mojo_backend"
    vec = PN.multigram_embed(b"hello world")
    assert vec.shape == (PN.multigram_dim(),)
    assert abs(float(np.linalg.norm(vec[: PN.UNIGRAM_DIM])) - 1.0) < 1e-6  # per-block L2, block 1


def test_quantize_i8_rounds_half_away_from_zero() -> None:
    """Pinned independently of the kernel: this rounding is what makes int8 scores match bit for bit."""
    src = np.array([0.5 / 127.0, -0.5 / 127.0, 1.0, -1.0, 2.0, -2.0, 0.0], np.float32)
    out = np.zeros(src.size, np.int8)
    PN.quantize_i8(src, 1, src.size, out)
    assert out.tolist() == [1, -1, 127, -127, 127, -127, 0]


def test_bm25_reference_skips_non_positive_scores() -> None:
    """The kernels emit only strictly positive documents, and so must the reference."""
    # Two docs share one term; a third shares nothing. Only the matching docs may appear.
    ids, scores = PN.bm25_csr_topk(
        post_docs=np.array([0, 1], np.int32),
        post_tfs=np.array([1, 1], np.int32),
        q_starts=np.array([0], np.int32),
        q_ends=np.array([2], np.int32),
        q_idf=np.array([0.7], np.float32),
        n_q=1,
        doc_denom=np.array([1.0, 1.0, 1.0], np.float32),
        n_docs=3,
        k1=1.8,
        k=10,
    )
    assert sorted(ids.tolist()) == [0, 1]
    assert all(s > 0.0 for s in scores)


def test_reference_empty_inputs_are_explicit() -> None:
    """Every entry point must answer an empty input with empty output, never an exception or a fabricated hit."""
    assert PN.flat_topk(np.zeros((0, 4), np.float32), 0, 4, np.zeros(4, np.float32), 5)[0].size == 0
    assert PN.bm25_csr_topk(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32),
                            np.zeros(0, np.int32), np.zeros(0, np.float32), 0,
                            np.zeros(0, np.float32), 0, 1.8, 5)[0].size == 0
    assert PN.bigram_embed(b"").sum() == 0.0
    assert PN.multigram_embed(b"").sum() == 0.0
    assert PN.code_tokenize_hashes("", set()) == []
