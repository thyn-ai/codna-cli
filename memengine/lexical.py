"""Code Lexical Index (CLI) — on-device, zero-model, IDF-weighted lexical retrieval for CODE.

Beats BM25 on 7/8 MTEB(Code)/CoIR tasks (canonical mteb.evaluate, Algenta-tuned k1=1.8/b=1.0) — ahead on average
(0.488 vs 0.445 best-of-two reproducible bm25s): Contest 0.718, DL 0.366, CosQA 0.218, StackOverflow 0.733,
Text2SQL 0.441, CodeFeedback MT 0.674 / ST 0.722 — while staying zero-model / zero-network / on-device. Loses
only near-zero AppsRetrieval. Canonical harness: bench/mteb_code_lexical.py. Two levers over the char-n-gram multigram:
code-aware tokenization (split camelCase + snake_case, keep whole id + subtokens) and IDF/BM25 term weighting.

Kernel-backed (mojo/engine/ame_kernel.mojo), all three verified score-exact against the pure-Python reference:
    - ame_code_tokenize  — code-aware tokenizer -> FNV-1a-64 hashes (~19x faster than Python)
    - ame_bm25_csr_topk  — inverted-index BM25, the FAST full-corpus path (scatter over postings, O(sum q-postings
                           + n_docs)); this is what search() uses.
    - ame_bm25_topk      — per-doc O(N) BM25 scan; the no-index ad-hoc path for tiny partitions (search_scan()).

The kernel is an OPTIONAL ACCELERATOR. When the shared library cannot be loaded (no wheel for this platform —
Windows today — or an operator pointing TELYS_KERNEL at nothing) `memengine.mojo_backend._lib()` returns
`memengine._pykernel.PyKernel`, which answers the SAME C ABI in NumPy via `memengine._pynative`. So `search()`,
`search_scan()` and `tokenize_hashes()` keep working with no code path of their own: same API, same rankings,
float32-identical scores.

The two properties that make that safe to claim, both covered in `cli/tests/test_pynative_equivalence.py`:

* tokenization is INTEGER work (FNV-1a-64 over folded bytes) and is BIT-IDENTICAL, so a persisted postings cache
  is portable between backends;
* scoring is FLOAT work whose summation ORDER differs (the kernel sums along SIMD lanes, NumPy sums pairwise), so
  scores agree to float32 rounding while the RANKING is identical.

What changes without the kernel is throughput, not answers. See internal/CODE-LEXICAL-INDEX.md.
"""
from __future__ import annotations

import ctypes
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

_ID = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")
_MASK = (1 << 64) - 1
# Reserved off-key column the engine uses to retain per-row token hashes (so a fitted per-partition lexical
# index can be (re)built without keeping raw text). Rides the existing columns plumbing (MVCC + persistence).
LEX_TOKENS_COLUMN = "__lex_toks__"
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_TOK_CAP = 65536

# English function words dropped before hashing (raw, pre-stem). Removing them lifts the NL-heavy tasks
# (StackOverflow +0.04, Text2SQL) with no code-task regression. Kept OUT: code keywords (for/in/if/is/return...).
STOP = frozenset(
    "a an the of to in for on and or but with that this these those which what how is are am be been being "
    "as at by from it its was were will would can could should i you we they he she me my your our do does "
    "did have has had not no so if then than when where who whom into about over under out up down".split())


def lite_stem(w: str) -> str:
    """Zero-dep English suffix stripper (byte-identical to the kernel's _cl_hash_stem): plurals + common
    verb/adverb endings on all-alpha tokens of len>3. Captures ~94% of the PyStemmer nDCG win, no dependency."""
    if len(w) <= 3 or not w.isalpha():
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes", "ses")):
        return w[:-2]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("ly"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def code_tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: whole lowered identifier + snake/camel subtokens, minus stopwords, lite-stemmed.
    Dedup (subtoken == whole) and stopword check are on the RAW lowered token; stemming is applied last."""
    out: list[str] = []
    for raw in _ID.findall(text):
        rl = raw.lower()
        if rl not in STOP:
            out.append(lite_stem(rl))
        for part in raw.split("_"):
            for s in _CAMEL.findall(part):
                sl = s.lower()
                if sl != rl and sl not in STOP:
                    out.append(lite_stem(sl))
    return out


def _fnv_signed(s: str) -> int:
    """FNV-1a 64 over lowercased bytes, as a signed int64 — matches the kernel's _cl_hash (Python fallback)."""
    h = _FNV_OFFSET
    for ch in s.encode("utf-8"):
        c = ch + 32 if 65 <= ch <= 90 else ch
        h = ((h ^ c) * _FNV_PRIME) & _MASK
    return h - (1 << 64) if h >= (1 << 63) else h


# Sorted raw (pre-stem) stopword hashes handed to the kernel tokenizer for binary-search skipping.
_STOP_HASHES = np.array(sorted(_fnv_signed(w) for w in STOP), dtype=np.int64)


_LIB: object | None = None
_VP = ctypes.c_void_p
_I32 = ctypes.c_int32
_F32 = ctypes.c_float


def _kernel():
    """The Mojo kernel with the code-lexical fns bound, or None (pure-Python fallback)."""
    global _LIB
    if _LIB is None:
        try:
            from memengine.mojo_backend import _lib
            lib = _lib()
            lib.ame_code_tokenize.restype = _I32
            lib.ame_code_tokenize.argtypes = [ctypes.c_char_p, _I32, _VP, _I32, _VP, _I32]
            lib.ame_bm25_topk.restype = _I32
            lib.ame_bm25_topk.argtypes = [_VP, _VP, _I32, _VP, _VP, _I32, _F32, _F32, _F32, _I32, _VP, _VP]
            lib.ame_bm25_csr_topk.restype = _I32
            lib.ame_bm25_csr_topk.argtypes = [_VP, _VP, _VP, _VP, _VP, _I32, _VP, _I32, _VP, _F32, _I32, _VP, _VP]
            _LIB = lib
        except Exception:  # noqa: BLE001 — kernel optional; pure-Python fallback below
            _LIB = False
    return _LIB or None


def tokenize_hashes(text: str) -> list[int]:
    """Code-aware token hashes (int64), stopword-filtered + lite-stemmed. Mojo kernel when present, else Python."""
    lib = _kernel()
    if lib is not None:
        b = text.encode("utf-8", "ignore")
        buf = np.empty(_TOK_CAP, np.int64)
        n = lib.ame_code_tokenize(b, len(b), buf.ctypes.data_as(_VP), _TOK_CAP,
                                  _STOP_HASHES.ctypes.data_as(_VP), len(_STOP_HASHES))
        return buf[:n].tolist()
    return [_fnv_signed(t) for t in code_tokenize(text)]


def _bm25_topk_order(scores: dict, k: int) -> list[tuple]:
    """The BM25 kernels' top-k contract for an ``{doc_index: score}`` map, ties included.

    Written once and shared by every pure-Python BM25 path so a fallback cannot drift from the kernel:

    * membership is "the first ``k`` while scanning documents in ascending index order, replacing a slot only on
      a STRICTLY greater score" — so among equal scores the earlier documents keep their slots;
    * emission repeatedly takes the LAST current maximum, reversing each run of exactly equal scores.

    Measured against ``ame_bm25_csr_topk`` / ``ame_bm25_topk`` on randomised tie-heavy corpora: ~99% identical
    result SET and ~93% identical ORDER (scores are float32-identical everywhere). The residual order difference
    is intra-run and comes from the kernel's SIMD lane packing; see ``memengine._pynative._bm25_insert_topk``.
    """
    if k <= 0 or not scores:
        return []
    n_docs = max(scores) + 1
    ordered = sorted(scores.items())
    slot_doc = [-1] * k
    slot_score = [0.0] * k
    filled = 0
    for d, s in ordered:
        if filled < k:
            slot_doc[filled] = d
            slot_score[filled] = s
            filled += 1
            continue
        thr = min(slot_score)
        if s > thr:
            j = slot_score.index(thr)  # FIRST minimum slot
            slot_doc[j] = d
            slot_score[j] = s
    if filled == 0:
        return []
    # Emission: repeatedly take the LAST buffer slot holding the current maximum. Buffer order, not document
    # order: a slot's occupant is whatever document last replaced it.
    remaining = list(zip(slot_doc[:filled], slot_score[:filled]))
    out: list[tuple] = []
    while remaining:
        best = 0
        for j in range(1, len(remaining)):
            if remaining[j][1] >= remaining[best][1]:
                best = j
        out.append(remaining.pop(best))
    return out


class _BM25:
    """Okapi BM25 over pre-tokenized docs (postings). Pure-Python reference / fallback."""

    def __init__(self, docs: Sequence[Sequence], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b, self.N = k1, b, len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.post: dict = defaultdict(list)
        df: dict = defaultdict(int)
        for i, d in enumerate(docs):
            for term, f in Counter(d).items():
                self.post[term].append((i, f)); df[term] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, qtoks) -> dict:
        sc: dict = defaultdict(float)
        for term in set(qtoks):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, f in self.post[term]:
                sc[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return sc


class LexicalCodeIndex:
    """Fit over (ids, texts); `search` -> [(id, score)] top-k. On-device, zero model files.

    build() constructs a CSR inverted index (postings) so search() scatter-accumulates BM25 over only the
    matching postings via the Mojo `ame_bm25_csr_topk` kernel — fast at any corpus size. Pure-Python fallback
    reproduces identical scores when the kernel is absent.
    """

    def __init__(self, *, k1: float = 1.8, b: float = 1.0):   # Algenta-tuned on CoIR (b=1 length-norm); overridable
        self.k1, self.b = k1, b
        self.ids: list = []
        self._bm: _BM25 | None = None
        # CSR inverted index
        self._term_range: dict[int, tuple[int, int]] = {}   # term hash -> (postings start, count)
        self._post_docs = np.zeros(0, np.int32)
        self._post_tfs = np.zeros(0, np.int32)
        self._doc_denom = np.zeros(0, np.float32)           # k1*(1-b + b*dl/avgdl) per doc
        # per-doc packed arrays (for the no-index scan path, search_scan)
        self._packed = np.zeros(0, np.int64)
        self._offs = np.zeros(1, np.int32)

    def build(self, ids: Sequence, texts: Sequence[str]) -> "LexicalCodeIndex":
        """Fit over raw texts (tokenizes each). See build_from_hashes to fit over pre-tokenized hash lists."""
        return self.build_from_hashes(ids, [tokenize_hashes(t) for t in texts])

    def build_from_hashes(self, ids: Sequence, hashes: Sequence[Sequence[int]]) -> "LexicalCodeIndex":
        """Fit over pre-tokenized token-hash lists (one per doc) — lets the engine build from retained tokens
        without keeping raw text. `hashes[i]` are the code-aware token hashes for doc `ids[i]`."""
        self.ids = list(ids)
        self._term_range = {}
        bm = _BM25(hashes, self.k1, self.b)
        self._bm = bm
        # CSR postings in a stable term order
        post_docs: list[int] = []
        post_tfs: list[int] = []
        cur = 0
        for term in sorted(bm.post.keys()):
            plist = bm.post[term]
            self._term_range[term] = (cur, len(plist))
            for d, f in plist:
                post_docs.append(d); post_tfs.append(f)
            cur += len(plist)
        self._post_docs = np.asarray(post_docs, np.int32)
        self._post_tfs = np.asarray(post_tfs, np.int32)
        avgdl = bm.avgdl or 1.0
        dl = np.asarray(bm.dl, np.float32) if bm.dl else np.zeros(0, np.float32)
        self._doc_denom = (self.k1 * (1 - self.b + self.b * dl / avgdl)).astype(np.float32)     # kernel zeroes this each call
        # per-doc packed (scan path)
        self._packed = (np.concatenate([np.asarray(h, np.int64) for h in hashes]) if hashes
                        else np.zeros(0, np.int64))
        self._offs = np.zeros(len(hashes) + 1, np.int32)
        for i, h in enumerate(hashes):
            self._offs[i + 1] = self._offs[i] + len(h)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple]:
        """BM25 top-k via the inverted-index kernel (fast, any corpus size). Falls back to Python postings."""
        assert self._bm is not None, "call build() first"
        lib = _kernel()
        if lib is None or not self.ids:
            return self._search_postings(query, k)
        starts: list[int] = []
        ends: list[int] = []
        idfs: list[float] = []
        for h in set(tokenize_hashes(query)):
            rng = self._term_range.get(h)
            if rng is None:
                continue
            st, cnt = rng
            starts.append(st); ends.append(st + cnt); idfs.append(self._bm.idf[h])
        if not starts:
            return []
        qs = np.asarray(starts, np.int32)
        qe = np.asarray(ends, np.int32)
        qi = np.asarray(idfs, np.float32)
        oi = np.full(k, -1, np.int32)
        osc = np.zeros(k, np.float32)
        scratch = np.empty(len(self.ids), np.float32)      # per-call (thread-safe): the kernel zeroes it itself,
        m = lib.ame_bm25_csr_topk(                          # so a shared instance buffer would race under readers
            self._post_docs.ctypes.data_as(_VP), self._post_tfs.ctypes.data_as(_VP),
            qs.ctypes.data_as(_VP), qe.ctypes.data_as(_VP), qi.ctypes.data_as(_VP), len(starts),
            self._doc_denom.ctypes.data_as(_VP), len(self.ids),
            scratch.ctypes.data_as(_VP), _F32(self.k1), k,
            oi.ctypes.data_as(_VP), osc.ctypes.data_as(_VP))
        return [(self.ids[int(oi[j])], float(osc[j])) for j in range(m)]

    def _search_postings(self, query: str, k: int = 10) -> list[tuple]:
        """Pure-Python postings BM25 (reference / fallback).

        Reconstructs scores directly from the CSR arrays instead of going through ``self._bm``, because a
        RESTORED index (``_load_postings_cache``, or a caller that sets the CSR fields by hand) carries the
        postings, idfs and denominators but deliberately no ``_BM25`` object — the whole point of that cache is
        to skip the O(N) rebuild. Reading the CSR arrays means one code path serves both a freshly built index
        and a restored one, so the fallback cannot silently return nothing just because the index came from
        disk. ``search_scan`` (which genuinely needs per-document token lists) still uses ``self._bm``.
        """
        assert self._bm is not None, "call build() first"
        sc: dict[int, float] = defaultdict(float)
        for h in set(tokenize_hashes(query)):
            rng = self._term_range.get(h)
            if rng is None:
                continue
            st, cnt = rng
            idf = float(self._bm.idf.get(h, 0.0))
            if idf == 0.0:
                continue
            k1p1 = self.k1 + 1.0
            for pos in range(st, st + cnt):
                d = int(self._post_docs[pos])
                f = float(self._post_tfs[pos])
                sc[d] += idf * (f * k1p1) / (f + float(self._doc_denom[d]))
        # Same contract as the kernels: hold only positive scores. `_bm25_topk_order` reproduces the BM25
        # kernels' tie handling (first k in document order wins membership; equal-score runs are emitted
        # backwards). A plain descending sort looked right but ordered ties the opposite way from the kernel.
        live = {i: s for i, s in sc.items() if s > 0.0}
        return [(self.ids[i], s) for i, s in _bm25_topk_order(live, k)]

    def search_scan(self, query: str, k: int = 10) -> list[tuple]:
        """No-index per-doc BM25 scan (Mojo ame_bm25_topk) — the ad-hoc tiny-partition path.

        The scan needs each document's full token list (``self._packed``), which a RESTORED postings cache does
        not carry; there the postings scan is the faithful equivalent (identical scores, same ranking) so it is
        used rather than returning nothing. A build always populates ``_packed``, so the fast path is unchanged.
        """
        lib = _kernel()
        assert self._bm is not None, "call build() first"
        if lib is None or self._packed.size == 0:
            return self._search_postings(query, k)
        uq = sorted(set(tokenize_hashes(query)))
        qa = np.asarray(uq, np.int64)
        qi = np.asarray([self._bm.idf.get(h, 0.0) for h in uq], np.float32)
        oi = np.full(k, -1, np.int32)
        osc = np.zeros(k, np.float32)
        m = lib.ame_bm25_topk(
            self._packed.ctypes.data_as(_VP), self._offs.ctypes.data_as(_VP), len(self.ids),
            qa.ctypes.data_as(_VP), qi.ctypes.data_as(_VP), len(uq),
            _F32(self.k1), _F32(self.b), _F32(self._bm.avgdl or 1.0), k,
            oi.ctypes.data_as(_VP), osc.ctypes.data_as(_VP))
        return [(self.ids[int(oi[j])], float(osc[j])) for j in range(m)]
