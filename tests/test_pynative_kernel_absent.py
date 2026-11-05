"""The engine must run with the native kernel ABSENT.

Every test here drives the real retrieval surface (``LexicalCodeIndex``, the Telys localizer) while the Mojo
kernel is unavailable, and asserts that it produces the SAME answers it produces with the kernel. That is the
whole Windows claim, made falsifiable on a machine that has a kernel: point the resolver at nothing and the code
path must not care.

WHY FORCE ABSENCE BY PATH rather than monkeypatching ``_kernel`` to return ``None``: patching the lookup would
prove only that this process can survive a stubbed function. Pointing ``TELYS_KERNEL`` at a nonexistent file
exercises the real resolution chain, the real ``PyKernel`` construction, the real one-time announcement, and the
real cache key — i.e. the path a Windows machine actually takes.

Note the ordering trap these tests exist to avoid: a module-level ``_LIB`` caches the resolved backend for the
process, so a test that installs the absence AFTER an earlier test already loaded the kernel would silently test
the kernel. Each test therefore reloads the memengine modules under the absence it wants to assert.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

telys = pytest.importorskip("telys", reason="the Telys SDK is not installed in the minimal offline suite")

# NO module-scope sys.path / sys.modules mutation here: pytest imports every test module during collection, so a
# module-level pin would leak into the whole session and change what unrelated tests import (that mistake briefly
# made eight unrelated memory tests "pass" by accident). All global mutation lives in the `_vendored_imports`
# fixture below, which restores everything.
_CLI_DIR = Path(__file__).resolve().parents[1]
MISSING_KERNEL = "/nonexistent/libame_kernel.dylib"
_KERNEL_ENV = ("TELYS_KERNEL", "AME_KERNEL", "CODNA_TELYS_KERNEL")


@pytest.fixture(autouse=True)
def _vendored_imports():
    """Pin `memengine` to the vendored copy for the duration of ONE test, then restore everything.

    Two hazards this handles, both observed on this machine:

    * ``telys.paths`` pushes the verified runtime's ``pysite`` to the front of sys.path, and that directory carries
      a ``memengine`` WITHOUT ``_pynative`` — so which engine a test exercises would otherwise depend on whether
      another test imported ``telys`` first;
    * the vendored engine only wins reliably while it is the first ``memengine`` on the path.

    Restoration matters as much as the pin: an unrestored path/sys.modules change silently alters every later test.
    """
    saved_path = list(sys.path)
    saved_modules = {
        name: mod for name, mod in list(sys.modules.items()) if name == "memengine" or name.startswith("memengine.")
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        sys.path[:] = [p for p in sys.path if "/.telys/runtime/" not in p.replace("\\", "/")]
        sys.path.insert(0, str(_CLI_DIR))
        importlib.invalidate_caches()
        yield
    finally:
        for name in [n for n in list(sys.modules) if n == "memengine" or n.startswith("memengine.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        importlib.invalidate_caches()


_KERNEL_ENV = ("TELYS_KERNEL", "AME_KERNEL", "CODNA_TELYS_KERNEL")


def _reload_under(kernel_path: str | None):
    """`memengine` loaded with the kernel resolvable at ``kernel_path`` (None = env var unset).

    Assumes the caller has already pinned the vendored copy (the autouse fixture does): ``_LIB_PATH`` is resolved at
    import time, so a reload is required for a changed env var to take effect at all.
    """
    if kernel_path is None:
        for name in _KERNEL_ENV:
            os.environ.pop(name, None)
    else:
        os.environ["TELYS_KERNEL"] = kernel_path
    for name in [n for n in list(sys.modules) if n == "memengine" or n.startswith("memengine.")]:
        del sys.modules[name]
    importlib.invalidate_caches()
    import memengine.lexical as lx
    import memengine.mojo_backend as mb

    return mb, lx


@pytest.fixture()
def _isolated_kernel_env(request, monkeypatch):
    """Run one test under a chosen kernel resolution, then RESTORE the ambient one.

    ``monkeypatch`` restores the ENVIRONMENT at teardown but knows nothing about module state, and the reload
    above rebuilds ``_LIB_PATH``/``_LIB`` from the environment it saw. Leaving those behind would be a
    cross-file landmine: every later test in the session would believe the kernel is missing, so
    ``test_pynative_equivalence`` — which decides at IMPORT time whether a native comparison is possible — would
    call into a ``None`` library and fail with an ``AttributeError`` instead of comparing (or skipping).
    Reloading the modules back under the restored environment makes each file independent of execution order.
    """
    saved = {name: os.environ.get(name) for name in _KERNEL_ENV}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _reload_under(None)


@pytest.fixture()
def no_kernel(_isolated_kernel_env):
    """memengine loaded with no native kernel available."""
    return _reload_under(MISSING_KERNEL)


@pytest.fixture()
def with_kernel(_isolated_kernel_env):
    """memengine loaded with whatever native kernel this machine actually has (skip if none)."""
    mb, lx = _reload_under(None)
    if mb.backend_kind() != "native":
        pytest.skip("this machine has no native Mojo kernel to compare against")
    return mb, lx


def test_backend_kind_reports_the_degradation(no_kernel) -> None:
    """Observable, not merely logged: a program can ask which backend answered."""
    mb, _lx = no_kernel
    assert mb.backend_kind() == "pynative"


def test_backend_kind_reports_native_when_the_kernel_loads(with_kernel) -> None:
    mb, _lx = with_kernel
    assert mb.backend_kind() == "native"


def test_degradation_is_announced_once(no_kernel, capsys) -> None:
    """The fallback must be DELIBERATE AND VISIBLE. A silent substitution was the documented failure mode."""
    mb, _lx = no_kernel
    mb._pykernel._warned[0] = False  # noqa: SLF001 - reset the once-per-process latch for this assertion
    mb._lib()  # noqa: SLF001
    mb._lib()  # noqa: SLF001 - a second use must NOT re-announce
    err = capsys.readouterr().err
    assert err.count("native AME kernel is not available") == 1


def test_lexical_search_works_and_ranks_correctly_without_the_kernel(no_kernel) -> None:
    _mb, lx = no_kernel
    docs = [
        "def lost_cookie_after_redirect():\n    return 'cookiejar'\n",
        "def unrelated_helper():\n    return 1\n",
        "class SessionStore:\n    def clear(self): pass\n",
    ]
    ids = ["d0", "d1", "d2"]
    index = lx.LexicalCodeIndex().build(ids, docs)
    hits = index.search("cookies lost after redirect", 3)
    assert hits, "the BM25 fallback must find a match, not return an empty result"
    assert hits[0][0] == "d0"
    assert all(score > 0.0 for _id, score in hits)


def test_lexical_search_and_scan_agree_without_the_kernel(no_kernel) -> None:
    """The postings path and the packed-scan path must agree; both are fallbacks for the same corpus."""
    _mb, lx = no_kernel
    docs = ["def reconcile_ledger(order): return order", "def migrate_schema(): pass", "cookiejar sessions"]
    index = lx.LexicalCodeIndex().build(["a", "b", "c"], docs)
    postings = index.search("reconcile ledger", 3)
    scan = index.search_scan("reconcile ledger", 3)
    assert [i for i, _ in postings] == [i for i, _ in scan]


def test_search_results_are_identical_with_and_without_the_kernel(no_kernel, with_kernel) -> None:
    """THE equivalence claim: same corpus, same query, same ranking and float32-equal scores.

    This is the test that would catch a behavioural fork — a fallback that "works" but ranks differently, which
    the brief calls out as worse than unavailability.
    """
    docs = [
        "def reconcile_ledger(order_refund): move the ledger entry after the migration",
        "def migrate_schema(): move columns to the new schema",
        "class LedgerEntry: pass",
        "cookie jar is lost after a redirect in the session store",
        "def unrelated(): return 0",
    ] * 3
    ids = [f"d{i}" for i in range(len(docs))]
    query = "reconcile ledger after migration"

    _mb_no, lx_no = no_kernel
    without = lx_no.LexicalCodeIndex().build(ids, docs).search(query, 5)

    _mb_yes, lx_yes = with_kernel
    with_ = lx_yes.LexicalCodeIndex().build(ids, docs).search(query, 5)

    assert [i for i, _ in without] == [i for i, _ in with_], "rankings diverged between backends"
    for (_i, a), (_j, b) in zip(without, with_):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-5)


def test_tokenizer_is_identical_without_the_kernel(no_kernel, with_kernel) -> None:
    """Token hashing is integer FNV-1a, so it must be bit-identical — this is what makes caches portable."""
    _mb_no, lx_no = no_kernel
    _mb_yes, lx_yes = with_kernel
    for text in ("def lost_cookie(): return cookiejar", "HTTPServer XMLHttpRequest", "A_B_C d.e-f 123abc"):
        assert lx_no.tokenize_hashes(text) == lx_yes.tokenize_hashes(text)


def _tie_heavy_corpus():
    """A corpus built to contain EXACT score ties, which is where a ranking fork would hide.

    Documents repeat a tiny vocabulary so several of them score bit-identically; a realistic corpus has far
    fewer ties, which is why the naive fallback looked correct without this test.
    """
    import random

    rng = random.Random(11)
    vocab = "alpha beta gamma delta epsilon".split()
    cases = []
    for _ in range(60):
        ndocs = rng.randint(2, 14)
        docs = [" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 3))) for _ in range(ndocs)]
        query = " ".join(rng.sample(vocab, 2))
        cases.append(([f"d{i}" for i in range(ndocs)], docs, query, rng.randint(1, ndocs)))
    return cases


def test_tie_heavy_rankings_agree_on_the_result_set_and_on_scores(no_kernel, with_kernel) -> None:
    """THE tie contract, asserted at the strength that is actually guaranteed.

    Exact-tie ORDER is not asserted as identical. The kernel's intra-tie order comes out of its SIMD lane packing
    and is only ~93% reproducible from outside (measured over randomised tie-heavy corpora); a test demanding 100%
    would be pinning a property the reference cannot hold, and would break on a kernel rebuild for no real reason.

    What IS asserted, and what a behavioural fork would break:

    * the returned SET is identical — same documents, or the fallback is retrieving different evidence;
    * when both return the same documents, their SCORES are float32-equal — same scoring function;
    * on the no-tie case, which is what real corpora overwhelmingly are, the ranking is identical point-for-point.
    """
    _mb_no, lx_no = no_kernel
    _mb_yes, lx_yes = with_kernel
    set_mismatches = []
    no_tie_cases = ties_seen = 0
    for ids, docs, query, k in _tie_heavy_corpus():
        without = lx_no.LexicalCodeIndex().build(ids, docs).search(query, k)
        with_ = lx_yes.LexicalCodeIndex().build(ids, docs).search(query, k)
        got_ids = [i for i, _ in without]
        want_ids = [i for i, _ in with_]
        if set(got_ids) != set(want_ids):
            set_mismatches.append((query, k, got_ids, want_ids))
            continue
        by_id = dict(with_)
        for doc_id, score in without:
            assert score == pytest.approx(by_id[doc_id], rel=1e-6, abs=1e-6), (
                f"score for {doc_id} differs: {score} vs {by_id[doc_id]}"
            )
        distinct = len({round(float(s), 9) for _, s in with_})
        if distinct == len(with_) and len(with_) > 1:
            no_tie_cases += 1
            assert got_ids == want_ids, f"no-tie ranking diverged for {query!r}"
        else:
            ties_seen += 1

    assert not set_mismatches[:3], f"tie-heavy result sets diverged (showing first 3): {set_mismatches[:3]}"
    # Guard against the corpus degenerating into "no ties at all", which would make this test vacuous.
    assert ties_seen > 0, "the corpus produced no exact ties; the tie contract went untested"
    assert no_tie_cases > 0


def test_empty_corpus_answers_empty_without_the_kernel(no_kernel) -> None:
    _mb, lx = no_kernel
    index = lx.LexicalCodeIndex().build([], [])
    assert index.search("anything", 5) == []


def test_partitioned_vector_index_builds_and_searches_without_the_kernel(no_kernel) -> None:
    """The vector engine is the other half of the optional-kernel claim; it must not need the library either."""
    mb, _lx = no_kernel
    assert mb.backend_kind() == "pynative"
    import memengine.embedding as emb_mod
    import memengine.partitioned as part_mod

    importlib.reload(emb_mod)
    importlib.reload(part_mod)
    embedder = emb_mod.AlgentaBigramEmbedder()
    index = part_mod.PartitionedVectorIndex(embedder.dim)
    texts = ["refund ledger reconcile", "migrate schema columns", "cookie jar redirect"]
    vectors = np.stack([embedder._one(t) for t in texts])  # noqa: SLF001
    index.build(vectors, np.arange(3), np.array(["a", "a", "b"], dtype=object), key_name="scope")
    ids, scores = index.search(embedder._one("refund reconcile"), 2)  # noqa: SLF001
    assert ids.shape == (2,)
    assert ids[0] == 0
    assert scores[0] >= scores[1]
