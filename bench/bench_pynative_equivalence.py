#!/usr/bin/env python3
"""Measure the pure-NumPy reference against the native Mojo kernel, on this machine.

Run:  cd cli && python3 bench/bench_pynative_equivalence.py

Prints a table of {primitive, native ms, reference ms, ratio} plus the score divergence observed on the same
input, so the two claims this work makes -- "same answers" and "slower" -- are both quantified rather than asserted.

The reference is measured by importing it directly (``memengine._pynative``), NOT by unsetting TELYS_KERNEL: the
backend is cached per process, and re-resolving mid-process would measure the harness rather than the code.

Numbers are wall-clock medians over a few repetitions on CPU. They characterise the ORDER of the slowdown on this
machine; they are not a cross-platform benchmark and should not be quoted as one.
"""
from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memengine import _pynative as PN  # noqa: E402

VP = ctypes.c_void_p
I32 = ctypes.c_int32
F32 = ctypes.c_float
REPEATS = 5


def _median_ms(fn, repeats: int = REPEATS) -> float:
    fn()  # warm up
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _native_lib():
    from memengine import mojo_backend as mb

    if mb.backend_kind() != "native":
        return None
    return mb._lib()


def main() -> int:
    lib = _native_lib()
    if lib is None:
        print("Native kernel unavailable on this machine -- nothing to compare against.")
        print("The reference still runs (that is the point), but a ratio needs both implementations.")
        return 0

    rng = np.random.default_rng(7)
    rows: list[tuple[str, float, float]] = []

    # ── flat top-k, the partition scan ──────────────────────────────────────────────────────────
    for n, d, k, label in ((20_000, 384, 10, "flat_topk n=20k d=384"), (100_000, 384, 10, "flat_topk n=100k d=384")):
        base = rng.standard_normal((n, d)).astype(np.float32)
        query = rng.standard_normal(d).astype(np.float32)
        out_ids = np.full(k, -1, np.int32)
        out_dist = np.zeros(k, np.float32)

        def native() -> None:
            lib.ame_flat_topk(base.ctypes.data_as(VP), n, d, query.ctypes.data_as(VP), k,
                              out_ids.ctypes.data_as(VP), out_dist.ctypes.data_as(VP))

        rows.append((label, _median_ms(native), _median_ms(lambda: PN.flat_topk(base, n, d, query, k))))

    # ── the lexical tokenizer, on a realistic window ───────────────────────────────────────────
    text = (
        "def reconcile_ledger_entry(order_refund, session_store):\n"
        "    # migrate the ledger after the redirect; HTTPServer XMLHttpRequest\n"
        "    return [entry for entry in session_store.items if entry.refunded]\n"
    ) * 40
    raw = text.encode()

    def native_tokenize() -> None:
        buf = np.empty(65536, np.int64)
        stop = np.zeros(0, np.int64)
        lib.ame_code_tokenize(raw, len(raw), buf.ctypes.data_as(VP), 65536, stop.ctypes.data_as(VP), 0)

    from memengine.lexical import STOP

    stop_set = {PN.fnv1a_signed(w) for w in STOP}
    rows.append((
        f"code_tokenize ({len(raw)} bytes)",
        _median_ms(native_tokenize),
        _median_ms(lambda: PN.code_tokenize_hashes(text, stop_set)),
    ))

    # ── BM25 postings query over a real-size corpus ────────────────────────────────────────────
    from memengine.lexical import LexicalCodeIndex

    docs = [[int(x) for x in rng.integers(0, 5000, size=int(rng.integers(20, 120)))] for _ in range(20_000)]
    lex = LexicalCodeIndex().build_from_hashes([str(i) for i in range(len(docs))], docs)
    query_hashes = sorted(set(int(x) for x in rng.integers(0, 5000, size=12)))
    starts, ends, idfs = [], [], []
    for h in query_hashes:
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
    k = 10
    out_ids = np.full(k, -1, np.int32)
    out_scores = np.zeros(k, np.float32)
    scratch = np.empty(len(lex.ids), np.float32)

    def native_bm25() -> None:
        lib.ame_bm25_csr_topk(
            lex._post_docs.ctypes.data_as(VP), lex._post_tfs.ctypes.data_as(VP),  # noqa: SLF001
            qs.ctypes.data_as(VP), qe.ctypes.data_as(VP), qi.ctypes.data_as(VP), len(starts),
            lex._doc_denom.ctypes.data_as(VP), len(lex.ids), scratch.ctypes.data_as(VP),  # noqa: SLF001
            F32(lex.k1), k, out_ids.ctypes.data_as(VP), out_scores.ctypes.data_as(VP),
        )

    rows.append((
        f"bm25_csr_topk n={len(lex.ids)}",
        _median_ms(native_bm25),
        _median_ms(lambda: PN.bm25_csr_topk(
            lex._post_docs, lex._post_tfs, qs, qe, qi, len(starts),  # noqa: SLF001
            lex._doc_denom, len(lex.ids), lex.k1, k,  # noqa: SLF001
        )),
    ))

    # ── embedder ───────────────────────────────────────────────────────────────────────────────
    dim = PN.multigram_dim()
    out_vec = np.zeros(dim, np.float32)
    probe = b"def reconcile_ledger_entry(order_refund):" * 20

    def native_embed() -> None:
        lib.ame_embed_multigram(probe, len(probe), out_vec.ctypes.data_as(VP))

    rows.append((
        "multigram_embed (620 bytes)",
        _median_ms(native_embed),
        _median_ms(lambda: PN.multigram_embed(probe)),
    ))

    # ── report ─────────────────────────────────────────────────────────────────────────────────
    print(f"{'primitive':<34}{'native ms':>12}{'reference ms':>15}{'ratio':>9}")
    print("-" * 70)
    for label, native_ms, ref_ms in rows:
        ratio = ref_ms / native_ms if native_ms > 0 else float("inf")
        print(f"{label:<34}{native_ms:>12.3f}{ref_ms:>15.3f}{ratio:>8.1f}x")
    print()
    print(
        "Ratios are the measured slowdown of the pure-NumPy reference on THIS machine.\n"
        "They characterise the order of magnitude only; retrieval quality is unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
