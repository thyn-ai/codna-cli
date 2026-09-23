"""Kernel-backed embedding providers (telys-runtime).

The embedding INTERFACES (EmbeddingProfile, EmbeddingProvider, CallableEmbedder) are the public contract and
live in the SDK (`telys.embedding`). This module ships the providers that call the native kernel, so they are
part of the closed runtime:
  - AlgentaBigramEmbedder : wraps ame_embed_bigram — a fast lexical/typo-tolerant embedder (no network); its
                            dimension is sourced from the kernel (ame_embed_bigram_dim), not hardcoded.
  - AlgentaPooledEmbedder : wraps ame_embed_neural — a Mojo-native POOLED TOKEN embedder (masked-mean of an
                            externally provided token-embedding table). NOT a transformer (no attention).

Re-exported here for backward-compatible imports: EmbeddingProfile/Provider/CallableEmbedder from telys.
"""
from __future__ import annotations

import ctypes
import hashlib

import numpy as np

from telys.embedding import CallableEmbedder, EmbeddingProfile, EmbeddingProvider  # noqa: F401  (re-export)
from memengine import _validate
from memengine.mojo_backend import _lib

__all__ = ["EmbeddingProvider", "EmbeddingProfile", "CallableEmbedder",
           "AlgentaBigramEmbedder", "AlgentaMultigramEmbedder", "AlgentaPooledEmbedder"]


class AlgentaBigramEmbedder(EmbeddingProvider):
    """Fast lexical bigram embedder (ame_embed_bigram). Typo-tolerant; weak semantics; zero network.

    The dimension is sourced ONCE from the kernel (ame_embed_bigram_dim → BIGRAM_DIM) — there is no hardcoded
    dim on the host side, so a kernel dim change propagates here automatically instead of OOB-writing.

    Zero-vector contract: empty / no-lexical-content input returns the all-zero vector (the kernel skips L2
    when the norm is 0). This is the one documented exception to normalization="l2"; it sorts last under inner
    product. Never L2-renormalize it (that would divide by zero)."""

    def __init__(self, *, max_bytes: int = 1 << 20) -> None:
        self.lib = _lib()
        self.dim = int(self.lib.ame_embed_bigram_dim())   # single source of truth (kernel BIGRAM_DIM)
        if self.dim <= 0:
            raise RuntimeError(f"kernel reported a non-positive bigram dim ({self.dim})")
        self.max_bytes = int(max_bytes)

    @property
    def profile(self) -> EmbeddingProfile:
        return EmbeddingProfile("algenta", "bigram", "1.0.0", self.dim, normalization="l2", distance="ip",
                                pooling="bigram-hash", tokenizer_hash="utf8-bytes")

    def _one(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"bigram embedder expects str, got {type(text).__name__}")
        b = text.encode("utf-8")
        if b"\x00" in b:                                   # embedded NUL: reject (deterministic, no silent corruption)
            raise ValueError("text contains an embedded NUL byte (U+0000) — not allowed")
        if len(b) > self.max_bytes:                        # deterministic prefix-truncate: a bigram histogram saturates
            b = b[: self.max_bytes]
        _validate.fits_i32(len(b), "text byte length")
        o = np.zeros(self.dim, np.float32)
        self.lib.ame_embed_bigram(b, len(b), o.ctypes.data_as(ctypes.c_void_p))
        assert o.shape[0] == self.dim                      # kernel must write exactly self.dim floats
        return o

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(np.stack([self._one(t) for t in texts]), np.float32)


class AlgentaMultigramEmbedder(EmbeddingProvider):
    """Lexical MULTIGRAM fusion embedder (ame_embed_multigram): one concatenated [unigram | bigram | trigram]
    vector with each block L2-normalised independently. Typo-tolerant; zero network; zero model artifacts.

    Per-block L2 is intentional: inner-product search then computes cos_u + cos_b + cos_t (equal-weight
    fusion) directly — and a consumer can reweight blocks by scaling the query sub-blocks (see block_dims).
    A Monte-Carlo source-localization study (bench/mc_ngram_localization.py) chose this over the plain bigram:
    on NL->code it beats bigram by ~+12-17 pts recall@10 and ties a 16 MB on-device semantic model on recall.

    Every dimension is sourced from the kernel (ame_embed_{unigram,bigram,trigram,multigram}_dim) — no hardcoded
    dims host-side. Zero-vector contract is PER BLOCK: a too-short input leaves that block all-zeros (so the
    full vector is NOT unit-norm; each non-empty block is). Never L2-renormalize the whole vector."""

    def __init__(self, *, max_bytes: int = 1 << 20) -> None:
        self.lib = _lib()
        self.dim = int(self.lib.ame_embed_multigram_dim())
        self.block_dims = (int(self.lib.ame_embed_unigram_dim()),
                           int(self.lib.ame_embed_bigram_dim()),
                           int(self.lib.ame_embed_trigram_dim()))   # (unigram, bigram, trigram) block widths
        if self.dim <= 0 or sum(self.block_dims) != self.dim:
            raise RuntimeError(f"kernel multigram layout inconsistent: dim={self.dim} blocks={self.block_dims}")
        self.max_bytes = int(max_bytes)

    @property
    def profile(self) -> EmbeddingProfile:
        # normalization="l2-blockwise": each [u|b|t] block is unit-norm, the concatenation is not. distance ip.
        return EmbeddingProfile("algenta", "multigram", "1.0.0", self.dim, normalization="l2-blockwise",
                                distance="ip", pooling="multigram-hash", tokenizer_hash="utf8-bytes")

    def _one(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"multigram embedder expects str, got {type(text).__name__}")
        b = text.encode("utf-8")
        if b"\x00" in b:
            raise ValueError("text contains an embedded NUL byte (U+0000) — not allowed")
        if len(b) > self.max_bytes:
            b = b[: self.max_bytes]
        _validate.fits_i32(len(b), "text byte length")
        o = np.zeros(self.dim, np.float32)
        self.lib.ame_embed_multigram(b, len(b), o.ctypes.data_as(ctypes.c_void_p))
        assert o.shape[0] == self.dim
        return o

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(np.stack([self._one(t) for t in texts]), np.float32)


class AlgentaPooledEmbedder(EmbeddingProvider):
    """Mojo-native POOLED TOKEN embedder (ame_embed_neural): masked-mean of a token-embedding table.

    NOT a transformer — no attention across the sequence. Pools an EXTERNALLY PROVIDED table (e.g. a model's
    input-embedding matrix exported to .bin); the runtime trains nothing. Provide table_path (.bin, row-major
    f32, vocab_size×emb_dim), vocab (token->id dict or path), and a tokenizer (callable text->list[str]).
    """

    def __init__(self, table_path: str, vocab, emb_dim: int, tokenizer, *, model_id: str = "pooled",
                 model_version: str = "0.0.0") -> None:
        import json
        self.lib = _lib()
        self.dim = int(emb_dim)
        if self.dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}")
        self.emb = np.ascontiguousarray(np.fromfile(table_path, dtype="<f4").reshape(-1, emb_dim), np.float32)
        self.rows = int(self.emb.shape[0])                 # vocab/table row count (for token-id bounds)
        if self.rows <= 0:
            raise ValueError(f"embedding table at {table_path} is empty")
        self.vocab = vocab if isinstance(vocab, dict) else json.loads(open(vocab).read())
        self.unk = int(self.vocab.get("<UNK>", self.vocab.get("<unk>", 1)))
        self.tok = tokenizer
        self._embp = self.emb.ctypes.data_as(ctypes.c_void_p)
        self._mid, self._ver = model_id, model_version
        self._thash = "sha256:" + hashlib.sha256(("|".join(sorted(self.vocab))).encode()).hexdigest()[:16]

    @property
    def profile(self) -> EmbeddingProfile:
        return EmbeddingProfile("algenta", self._mid, self._ver, self.dim, normalization="l2", distance="ip",
                                pooling="masked-mean", tokenizer_hash=self._thash)

    def _one(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"pooled embedder expects str, got {type(text).__name__}")
        ids = [self.vocab.get(t, self.unk) for t in self.tok(text)] or [self.unk]
        # Clamp every token id into [0, rows) BEFORE the call: an out-of-range id (vocab/table mismatch)
        # would OOB-read the embedding table in the kernel (the Mojo side clamps too, as layer 2).
        a = np.clip(np.asarray(ids, np.int64), 0, self.rows - 1).astype(np.int32)
        _validate.fits_i32(len(a), "token count")
        o = np.zeros(self.dim, np.float32)
        self.lib.ame_embed_neural(self._embp, self.dim, self.rows, a.ctypes.data_as(ctypes.c_void_p), len(a),
                                  o.ctypes.data_as(ctypes.c_void_p))
        assert o.shape[0] == self.dim
        return o

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(np.stack([self._one(t) for t in texts]), np.float32)
