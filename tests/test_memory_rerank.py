"""Query-aware soft test-rerank in codna.memory (lossless recall-time policy).

Tests stay FULLY indexed; when the issue isn't test-oriented, test symbols are softly
down-weighted so source files surface — nothing is ever removed, so a bug IN a test
(flaky assertion, broken fixture, the --from-junit flow) stays localizable. The policy
was validated end-to-end by a Monte-Carlo source-localization eval over 900 sampled
queries (recall@10 0.67->0.75, MRR 0.30->0.41, non-overlapping 95% CIs, no regression);
these unit tests pin the policy logic and need no native kernel.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from codna import codeunits as CU
from codna.memory import (
    CodeMemory,
    CodeMemoryError,
    _is_test_symbol,
    _query_is_test_oriented,
    _soft_test_downweight,
)
from codna.codeunits import PythonExtractor


@pytest.mark.parametrize("q", [
    "the failing test asserts the wrong status",
    "a flaky pytest fixture in conftest",
    "unittest assertion error on redirect",
    "regression test for cookies",
])
def test_test_oriented_queries_detected(q):
    assert _query_is_test_oriented(q) is True


@pytest.mark.parametrize("q", [
    "the browser receives an incorrect file type label on download",
    "session loses cookies after a redirect",
    "a required flag can be omitted without error",
])
def test_source_queries_not_flagged_as_test(q):
    assert _query_is_test_oriented(q) is False


@pytest.mark.parametrize("symbol_type,path,expected", [
    ("test", None, True),
    ("test", "src/app.py", True),
    ("function", "pkg/tests/test_x.py", True),
    ("function", "pkg/foo_test.py", True),
    ("function", "conftest.py", True),
    ("function", "src/helpers.py", False),
    ("class", "src/sessions.py", False),
    (None, None, False),
])
def test_is_test_symbol(symbol_type, path, expected):
    assert _is_test_symbol(symbol_type, path) is expected


META = {
    "test1": {"symbol_type": "test", "path": "tests/test_helpers.py"},
    "src1": {"symbol_type": "function", "path": "src/helpers.py"},
}


def test_source_query_lifts_source_above_higher_scored_test():
    # test scores higher (0.90) than source (0.85); the soft prior must reorder for a non-test query
    out = _soft_test_downweight("download wrong mimetype", [("test1", 0.90), ("src1", 0.85)], META)
    assert [i for i, _ in out] == ["src1", "test1"]


def test_test_query_preserves_order_lossless():
    # test-oriented query: nothing is down-weighted, original order kept (and nothing removed)
    out = _soft_test_downweight("the failing test assertion", [("test1", 0.90), ("src1", 0.85)], META)
    assert [i for i, _ in out] == ["test1", "src1"]


@pytest.mark.parametrize("q", ["download bug", "the failing test assertion"])
def test_nothing_is_ever_removed(q):
    cands = [("test1", 0.90), ("src1", 0.85)]
    out = _soft_test_downweight(q, cands, META)
    assert {i for i, _ in out} == {"test1", "src1"}  # lossless: same set, only order can change


def test_downweight_disabled_via_env(monkeypatch):
    monkeypatch.setenv("CODNA_MEMORY_TEST_DOWNWEIGHT", "1.0")
    out = _soft_test_downweight("download bug", [("test1", 0.90), ("src1", 0.85)], META)
    assert [i for i, _ in out] == ["test1", "src1"]  # identity when prior is disabled


def test_stronger_downweight_still_lossless(monkeypatch):
    monkeypatch.setenv("CODNA_MEMORY_TEST_DOWNWEIGHT", "0.1")
    out = _soft_test_downweight("download bug", [("test1", 0.90), ("src1", 0.85)], META)
    assert [i for i, _ in out] == ["src1", "test1"]
    assert {i for i, _ in out} == {"test1", "src1"}


class FakeCollection:
    def __init__(self, result=None) -> None:
        self.where = None
        self.result = result or {"ids": [], "scores": [], "metadata": [], "explain": {}}

    def search_text(self, _query, *, top_k, where, explain, target_recall, with_metadata):
        self.where = where
        return self.result


def _memory_with_fake_collection(collection: FakeCollection) -> CodeMemory:
    mem = CodeMemory.__new__(CodeMemory)
    mem.repo_id = "repo"
    mem.service = None
    mem._scope_key = lambda repo_id, service, language: f"{repo_id}:{service}:{language}"
    mem._collection = lambda: collection
    return mem


def test_service_only_recall_filters_across_all_languages():
    collection = FakeCollection()
    mem = _memory_with_fake_collection(collection)

    mem.recall("latency", service="payments")

    assert collection.where == {"service": "payments"}


def test_service_and_language_recall_uses_exact_partition():
    collection = FakeCollection()
    mem = _memory_with_fake_collection(collection)

    mem.recall("latency", service="payments", language="go")

    assert collection.where == {"scope_key": "repo:payments:go"}


def test_language_only_recall_uses_default_service_partition():
    collection = FakeCollection()
    mem = _memory_with_fake_collection(collection)

    mem.recall("latency", language="rust")

    assert collection.where == {"scope_key": "repo::rust"}


def test_path_only_recall_filters_by_path():
    collection = FakeCollection()
    mem = _memory_with_fake_collection(collection)

    mem.recall("latency", path="src/app.go")

    assert collection.where == {"path": "src/app.go"}


def test_service_and_path_recall_combines_filters():
    collection = FakeCollection()
    mem = _memory_with_fake_collection(collection)

    mem.recall("latency", service="payments", path="src/app.go")

    assert collection.where == {"path": "src/app.go"}


def test_language_and_path_recall_combines_partition_and_path():
    collection = FakeCollection()
    mem = _memory_with_fake_collection(collection)

    mem.recall("latency", language="rust", path="src/lib.rs")

    assert collection.where == {"scope_key": "repo::rust"}


def test_service_language_and_path_recall_combines_partition_and_path():
    collection = FakeCollection()
    mem = _memory_with_fake_collection(collection)

    mem.recall("latency", service="payments", language="go", path="svc/main.go")

    assert collection.where == {"scope_key": "repo:payments:go"}


def test_secondary_recall_filters_are_applied_after_single_telys_filter(monkeypatch):
    monkeypatch.setenv("CODNA_MEMORY_RERANK", "lexical")
    collection = FakeCollection(
        result={
            "ids": ["match", "wrong-service", "wrong-path"],
            "scores": [0.9, 0.8, 0.7],
            "metadata": [
                {"service": "payments", "path": "src/app.go"},
                {"service": "billing", "path": "src/app.go"},
                {"service": "billing", "path": "src/app.go"},
            ],
            "explain": {},
        }
    )
    mem = _memory_with_fake_collection(collection)

    result = mem.recall("latency", service="payments", path="src/app.go")

    assert collection.where == {"path": "src/app.go"}
    assert [symbol["id"] for symbol in result["symbols"]] == ["match"]
    assert result["candidate_count"] == 1


def test_path_filter_is_applied_after_partition_filter(monkeypatch):
    monkeypatch.setenv("CODNA_MEMORY_RERANK", "lexical")
    collection = FakeCollection(
        result={
            "ids": ["match", "wrong-path"],
            "scores": [0.9, 0.8],
            "metadata": [
                {"service": "payments", "path": "src/app.go"},
                {"service": "payments", "path": "src/other.go"},
            ],
            "explain": {},
        }
    )
    mem = _memory_with_fake_collection(collection)

    result = mem.recall("latency", service="payments", language="go", path="src/app.go")

    assert collection.where == {"scope_key": "repo:payments:go"}
    assert [symbol["id"] for symbol in result["symbols"]] == ["match"]
    assert result["candidate_count"] == 1


def test_index_rejects_unsupported_language_before_opening_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(CU, "_EXTRACTORS", {".py": PythonExtractor()})
    mem = CodeMemory.__new__(CodeMemory)
    mem.repo_path = str(tmp_path)
    mem.repo_id = "repo"
    mem._collection = lambda: pytest.fail("unsupported language must fail before collection access")

    with pytest.raises(CodeMemoryError) as excinfo:
        mem.index(languages=("go",))

    assert "unsupported language extractor(s): go" in str(excinfo.value)


class FakeIndexUnit:
    id = "repo:a.py:target"
    text = "def target():\n    return 1\n"
    service = ""
    language = "python"

    def metadata(self, repo_id, scope, *, commit_sha, branch):
        return {
            "repo_id": repo_id,
            "scope_key": scope,
            "commit_sha": commit_sha,
            "branch": branch,
            "path": "a.py",
            "language": self.language,
            "symbol_type": "function",
            "qualname": "target",
            "service": self.service,
        }


class FakeIndexCollection:
    def __init__(self) -> None:
        self.deleted = []
        self.upserts = []
        self.compacted = False
        self.saved = False

    def ids(self, *, where):
        raise AssertionError(f"index prune must not depend on Telys ids(where={where!r})")

    def delete(self, ids):
        self.deleted.extend(ids)

    def upsert_texts(self, texts, ids, metadata):
        self.upserts.append((texts, ids, metadata))

    def compact(self):
        self.compacted = True

    def save(self):
        self.saved = True


def test_index_prune_uses_scope_manifest_not_telys_ids(tmp_path, monkeypatch):
    unit = FakeIndexUnit()
    collection = FakeIndexCollection()
    old_current = "repo:old.py:gone"
    old_foreign = "other:b.py:keep"
    previous_manifest = {
        old_current: {
            "hash": "old",
            "repo_id": "repo",
            "service": "",
            "language": "python",
            "scope_key": "repo::python",
        },
        old_foreign: {
            "hash": "foreign",
            "repo_id": "other",
            "service": "",
            "language": "python",
            "scope_key": "other::python",
        },
    }
    monkeypatch.setattr("codna.memory.resolve_languages", lambda languages: ("python",))
    monkeypatch.setattr(
        "codna.memory.extract_repo",
        lambda _repo_path, _repo_id, *, languages: ([unit], SimpleNamespace(files=1, skipped=0)),
    )
    monkeypatch.setattr("codna.memory._git", lambda *_args: "")

    mem = CodeMemory.__new__(CodeMemory)
    mem.repo_path = str(tmp_path)
    mem.repo_id = "repo"
    mem.service = None
    mem.branch = None
    mem.db_path = str(tmp_path / "db")
    mem._collection = lambda: collection
    mem._scope_key = lambda repo_id, service, language: f"{repo_id}:{service}:{language}"
    mem._embedder = SimpleNamespace(profile=SimpleNamespace(space_id=lambda: "fake-space"))
    mem._load_hashes = lambda: previous_manifest
    mem._save_hashes = lambda hashes: setattr(mem, "_saved_hashes", hashes)
    mem._write_meta = lambda **kwargs: setattr(mem, "_wrote_meta", kwargs)

    report = mem.index(languages=("python",))

    assert collection.deleted == [old_current]
    assert old_current not in mem._saved_hashes
    assert old_foreign in mem._saved_hashes
    assert mem._saved_hashes[unit.id]["scope_key"] == "repo::python"
    assert mem._saved_hashes[unit.id]["hash"]
    assert collection.upserts and collection.upserts[0][1] == [unit.id]
    assert collection.compacted is True and collection.saved is True
    assert report["indexed"] == 1 and report["removed"] == 1
