"""Native runtime compact serve artifacts: capability declaration + native/local equivalence.

`NativeRuntime.export_compact` used to raise `NotImplementedError` while every other runtime surface worked,
so `codna memory export` worked only under `TELYS_RUNTIME=local`. The Telys recall fix made the native
runtime genuinely reachable and the export test started running — unmasking the gap.

These tests pin the two properties that make the implementation trustworthy:

  * EQUIVALENCE — the same corpus exported by both runtimes produces byte-identical artifacts (every file
    plus `manifest.json`). That is the strong statement, and it holds because the native index carries
    everything the compact tier needs: the visible f32 base (`ty_index_export_base`), the logical row ids,
    the partition directory (contiguous physical layout) and the interned provenance columns
    (`ty_index_export_rowmeta`). The int8 slab uses the same rounding rule as the local sealer, and PQ
    codes come from the same faiss call, so no per-runtime difference is left to explain.
  * CAPABILITY — `supports_compact_export` is a real flag on the runtime, because `hasattr(collection,
    "export_compact")` is true even when the method raises (it is part of the SDK's RuntimeHandle
    contract). That wrong probe is what made the product test fail inside its own body.

Skips cleanly where the Telys surface is unavailable (offline CI lane), same posture as the memory suites.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os

import pytest

ARTIFACT_FILES = ("manifest.json", "rowids.i64", "vectors.i8")


def _write_repo(repo, *, n_files=3, n_funcs=4):
    os.makedirs(repo, exist_ok=True)
    for i in range(n_files):
        with open(os.path.join(repo, f"m{i}.py"), "w") as f:
            f.write("\n".join(
                f"def func_{i}_{j}(x):\n    '''thing {i} {j}'''\n    return x + {j}"
                for j in range(n_funcs)
            ))


def _code_memory(monkeypatch, repo, db):
    monkeypatch.setenv("CODNA_MEMORY_EMBED", "local")
    try:
        from codna.memory import CodeMemory
        return CodeMemory(repo, db_path=db)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"telys/kernel unavailable: {type(exc).__name__}: {exc}")


def _runtime(mem):
    return getattr(getattr(mem, "_eng", None), "_runtime", None)


def _runtime_name(mem) -> str:
    return type(_runtime(mem)).__name__


def test_native_runtime_declares_compact_export_capability(monkeypatch, tmp_path):
    """The capability flag is the honest probe: it is True only where the export really works."""
    repo, db = str(tmp_path / "repo"), str(tmp_path / "db")
    _write_repo(repo)
    mem = _code_memory(monkeypatch, repo, db)
    runtime = _runtime(mem)

    assert runtime is not None, "the resolved engine exposes no runtime handle"
    assert getattr(runtime, "supports_compact_export", None) is True, (
        f"{_runtime_name(mem)} must declare supports_compact_export=True"
    )
    # The method's presence is NOT the signal (it exists on every runtime) — assert both, so the test
    # documents why the flag is the one that matters.
    assert hasattr(mem._collection(), "export_compact")


def test_native_export_passes_the_product_contract(monkeypatch, tmp_path):
    """`codna memory export` produces a real artifact under the native runtime."""
    repo, db = str(tmp_path / "repo"), str(tmp_path / "db")
    _write_repo(repo)
    mem = _code_memory(monkeypatch, repo, db)

    info = mem.export_serve_artifact(str(tmp_path / "artifact"))

    assert info["documents"] == 12
    assert info["mode"] == "int8" and info["size_bytes"] > 0
    for name in ARTIFACT_FILES:
        assert os.path.isfile(os.path.join(info["artifact"], name)), name

    manifest = json.load(open(os.path.join(info["artifact"], "manifest.json")))
    assert manifest["compact_primary"] == "int8"
    assert manifest["n"] == 12 and manifest["dtype"] == "int8"
    # The compact tier drops the f32 base and the external id map — never write them.
    assert not os.path.exists(os.path.join(info["artifact"], "vectors.f32"))
    assert not os.path.exists(os.path.join(info["artifact"], "ty_base.npy"))
    # Full provenance travels with the artifact.
    for col in manifest["columns"]:
        assert os.path.isfile(os.path.join(info["artifact"], f"col_{col}.npy")), col


def test_export_requires_a_folded_int8_index(monkeypatch, tmp_path):
    """A wrong-shaped export must fail loudly, never write a partial artifact."""
    repo, db = str(tmp_path / "repo"), str(tmp_path / "db")
    _write_repo(repo)
    mem = _code_memory(monkeypatch, repo, db)
    runtime, col = _runtime(mem), mem._collection()
    mem.index()

    with pytest.raises(ValueError, match="unknown compact mode"):
        runtime.export_compact(col.idx, str(tmp_path / "bad-mode"), "f32")

    if getattr(col, "dtype", "f32") != "int8":  # an f32 base has no int8 slab to keep
        with pytest.raises(ValueError, match="dtype='int8'"):
            runtime.export_compact(col.idx, str(tmp_path / "f32"), "int8")


def test_i8_quantizer_matches_the_kernel_byte_for_byte():
    """The export's int8 slab must be the kernel's own quantization, not an approximation of it.

    Exact .5 boundaries are included deliberately: the kernel rounds HALF-AWAY-FROM-ZERO (a +/-0.5 nudge
    then truncation), which NumPy's `rint`/`round` would get wrong. That is the only place the two
    runtimes' slabs could drift, so it is pinned directly.
    """
    import numpy as np

    # Both halves need the SDK: `memengine.native_runtime` imports `telys.runtime` at module scope. The
    # offline CI lane ships no SDK, and an unimportable module is a genuine skip here — nothing about the
    # quantization rule is being hidden by it.
    try:
        from memengine.native_runtime import _quantize_i8
        from memengine.mojo_backend import _lib as kernel_lib
        lib = kernel_lib()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Telys SDK / Mojo kernel unavailable: {type(exc).__name__}: {exc}")

    rng = np.random.default_rng(3)
    unit = rng.standard_normal((64, 32)).astype(np.float32)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    boundary = np.array([[0.5 / 127, -0.5 / 127, 1.5 / 127, -1.5 / 127] * 8], np.float32)

    for name, vecs in (("random unit rows", unit), ("half boundaries", boundary)):
        n, dim = vecs.shape
        out = np.empty(n * dim, np.int8)
        src = np.ascontiguousarray(vecs, np.float32)
        lib.ame_quantize_i8(
            src.ctypes.data_as(ctypes.c_void_p), n, dim, out.ctypes.data_as(ctypes.c_void_p)
        )
        theirs = _quantize_i8(src, dim)
        assert np.array_equal(out, theirs), (
            f"{name}: export quantizer diverges from the kernel — "
            f"{np.flatnonzero(out != theirs)[:8].tolist()}"
        )


def test_partition_directory_rejects_a_non_contiguous_export(monkeypatch, tmp_path):
    """A non-contiguous partition layout must fail closed, never write a directory that lies.

    `[start, len]` only describes the right rows when each partition key occupies ONE contiguous run. The
    native base is contiguous by construction, but the export must not depend on that silently: if a key
    reappeared after its slice closed, `[start, len]` would point a reader at other partitions' rows.
    """
    import numpy as np

    repo, db = str(tmp_path / "repo"), str(tmp_path / "db")
    _write_repo(repo)
    mem = _code_memory(monkeypatch, repo, db)
    runtime = _runtime(mem)
    import memengine.native_runtime as native  # same module instance the runtime uses

    # A stand-in handle whose visible rows interleave two partition keys: a, b, a.
    class _Layout:
        dtype = "int8"
        dim = 4
        key_name = "scope"
        col_names: list = []
        lexical_tokens: dict = {}
        exact_crossover = 1 << 62
        embedder = None
        key_i2v: dict = {}
        _rw = native._RWLock()  # export_compact runs under the index write lock

        def __init__(self):
            self._h = 0

    layout = _Layout()
    interleaved = (
        np.zeros((3, 4), np.float32),
        np.array([0, 1, 2], np.int64),
        ["a", "b", "a"],
        {},
    )
    monkeypatch.setattr(native.NativeRuntime, "_materialize_visible", lambda self, index: interleaved)
    monkeypatch.setattr(native, "_lib", lambda: _StubLib())

    artifact = str(tmp_path / "artifact")
    with pytest.raises(RuntimeError, match="not partition-contiguous"):
        runtime.export_compact(layout, artifact, "int8")
    assert not os.path.exists(os.path.join(artifact, "manifest.json"))


class _StubLib:
    """Just the three native entry points export_compact touches, for the layout test above."""

    def ty_index_visible_count(self, _handle):
        return 3

    def ty_index_mutated(self, _handle):
        return 0

    def ty_index_delta_count(self, _handle):
        return 0

    def ty_index_snapshot(self, _handle):
        return 0


def _runtimes():
    """Both runtime implementations, constructed in THIS process (no env switch, no session bleed)."""
    try:
        from memengine.native_runtime import NativeRuntime
        from memengine.runtime import LocalRuntime
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Telys runtime modules unavailable: {type(exc).__name__}: {exc}")
    return {"local": LocalRuntime(), "native": NativeRuntime()}


def _embedder():
    from memengine.embedding import EmbeddingProfile

    class _Fixed:
        """Deterministic provider: the artifact comparison must not depend on a model's batching."""

        profile = EmbeddingProfile(
            provider="test", model_id="fixed", model_version="1.0.0", dimension=16,
            normalization="l2", distance="ip", pooling="none", tokenizer_hash="utf8-bytes",
        )

        def embed_documents(self, texts):
            import numpy as np

            out = np.empty((len(texts), 16), np.float32)
            for i, text in enumerate(texts):
                seed = hashlib.sha256(text.encode()).digest()
                row = np.frombuffer(seed[:16], dtype=np.uint8).astype(np.float32) - 127.5
                out[i] = row / (np.linalg.norm(row) or 1.0)
            return out

        def embed_queries(self, texts):
            return self.embed_documents(texts)

    return _Fixed()


def _export_with(runtime, path) -> dict:
    """Build one collection through `runtime` and seal a compact artifact; return {file: sha256}."""
    from telys import Telys

    engine = Telys(os.path.join(path, "engine"), runtime=runtime)
    col = engine.create_collection(
        "c", 16, "scope", embedder=_embedder(), dtype="int8", filter_columns=("path", "kind"),
        lexical=True,
    )
    ids, keys, meta, texts = [], [], [], []
    for partition in ("a", "b"):
        for i in range(6):
            ids.append(f"{partition}-{i}")
            keys.append(partition)
            texts.append(f"def {partition}_{i}(x): return x + {i}")
            meta.append({"scope": partition, "path": f"{partition}/{i}.py", "kind": "function"})
    col.add_texts(texts, ids, meta)
    col.compact()  # fold: the compact tier requires a sealed generation, no pending delta
    col.save()
    artifact = col.export_compact(os.path.join(path, "artifact"))
    return {
        os.path.relpath(os.path.join(root, name), artifact): hashlib.sha256(
            open(os.path.join(root, name), "rb").read()
        ).hexdigest()
        for root, _dirs, files in os.walk(artifact)
        for name in files
    }


def test_native_and_local_export_are_byte_identical(tmp_path):
    """The same corpus through both runtimes yields byte-identical artifacts, manifest included."""
    runtimes = _runtimes()
    local = _export_with(runtimes["local"], str(tmp_path / "local"))
    native = _export_with(runtimes["native"], str(tmp_path / "native"))

    assert local, "the local runtime wrote no files"
    assert set(local) == set(native), (
        f"file sets differ:\n  local-only: {sorted(set(local) - set(native))}\n"
        f"  native-only: {sorted(set(native) - set(local))}"
    )
    differing = sorted(name for name in local if local[name] != native[name])
    assert not differing, f"byte-differing artifact files: {differing}"
