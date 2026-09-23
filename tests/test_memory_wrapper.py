"""Kernel-gated tests for the CodeMemory wrapper (need the Telys kernel + the [memory] extra).

Skipped cleanly when Telys isn't installed or Codna cannot resolve a local Telys kernel.
"""
from __future__ import annotations

import os

import pytest


def _telys_ready():
    try:
        from codna.memory import _configure_telys_kernel_env

        _configure_telys_kernel_env()
        from telys import scope_key  # noqa: F401
        from telys.embedding import AlgentaBigramEmbedder
        AlgentaBigramEmbedder()       # loads the kernel dylib; raises if absent
        return True
    except Exception:
        return False


if not _telys_ready():
    pytest.skip("Telys not installed or Codna could not resolve a local kernel", allow_module_level=True)

from telys import scope_key  # noqa: E402

from codna.memory import CodeMemory, HASH_FILENAME  # noqa: E402

MINI = os.path.join(os.path.dirname(__file__), "fixtures", "mini_repo")


def _mem(tmp_path, service="payments"):
    return CodeMemory(MINI, db_path=str(tmp_path / "db"), service=service)


def test_index_then_recall_is_partition_aware(tmp_path):
    mem = _mem(tmp_path)
    report = mem.index()
    assert report["indexed"] >= 4 and report["skipped"] == 1   # broken.py skipped, real symbols indexed
    assert "python" in report["languages"]

    hit = mem.recall("reconcile the ledger entry after an account migration", service="payments", language="python")
    assert hit["symbols"], "expected at least one recalled symbol"
    # scoped query -> partition-aware plan, on the exact scope we indexed into
    assert hit["explain"]["plan"].startswith("Partition")
    assert hit["explain"]["partition_value"] == scope_key(mem.repo_id, "payments", "python")
    # provenance is recovered from the stable id
    assert all(s["path"] and s["symbol_type"] for s in hit["symbols"])
    assert any("reconcile" in s["id"] or "process" in s["id"] for s in hit["symbols"])


def test_scope_isolation_no_leak_across_scopes(tmp_path):
    mem = _mem(tmp_path, service="payments")
    mem.index()
    # querying a DIFFERENT scope (service=billing) must not leak the payments-scoped rows
    miss = mem.recall("reconcile the ledger entry after an account migration", service="billing", language="python")
    assert miss["symbols"] == []
    assert miss["explain"]["partition_value"] == scope_key(mem.repo_id, "billing", "python")


def test_offkey_path_filter_is_honest_scatter_fallback(tmp_path):
    mem = _mem(tmp_path)
    mem.index()
    off = mem.recall("ledger", path="refund.py")
    assert off["explain"]["plan"] == "ScatterGatherExact"
    assert off["explain"].get("fallback") is True


def test_persist_and_reopen(tmp_path):
    db = str(tmp_path / "db")
    CodeMemory(MINI, db_path=db, service="payments").index()
    # a brand-new instance (fresh engine) over the same dir must still recall
    reopened = CodeMemory(MINI, db_path=db, service="payments")
    assert not reopened.is_empty()
    again = reopened.recall("reconcile ledger after migration", service="payments")
    assert again["symbols"]


def test_status_and_reset(tmp_path):
    db = str(tmp_path / "db")
    mem = CodeMemory(MINI, db_path=db, service="payments")
    mem.index()
    st = mem.status()
    assert st["indexed"] is True and st["engine"] == "telys"
    assert st["partition_key"] == "scope_key" and (st["documents"] or 0) >= 4
    assert st["embedding_space"]   # space_id pinned

    mem.reset()
    assert CodeMemory(MINI, db_path=db, service="payments").is_empty()


def test_cross_repo_shared_collection_no_clobber(tmp_path):
    """Two repos in ONE shared collection: (re-)indexing one must not delete the other's symbols."""
    db = str(tmp_path / "shared")
    a = tmp_path / "repoA"
    a.mkdir()
    (a / "a.py").write_text("def alpha_refund(order):\n    '''reconcile the ledger for a refund'''\n    return order\n")
    b = tmp_path / "repoB"
    b.mkdir()
    (b / "b.py").write_text("def beta_invoice(row):\n    '''export an invoice to csv'''\n    return row\n")

    CodeMemory(str(a), db_path=db).index()
    CodeMemory(str(b), db_path=db).index()      # indexing B must NOT prune A's scope
    CodeMemory(str(a), db_path=db).index()      # re-indexing A must NOT prune B's scope

    ra = CodeMemory(str(a), db_path=db).recall("reconcile ledger refund", language="python")
    rb = CodeMemory(str(b), db_path=db).recall("export invoice csv", language="python")
    assert ra["symbols"] and all(":a.py:" in s["id"] for s in ra["symbols"])   # A intact + scope-isolated
    assert rb["symbols"] and all(":b.py:" in s["id"] for s in rb["symbols"])   # B survived B-then-A indexing
    assert (CodeMemory(str(a), db_path=db).status()["documents"] or 0) >= 2     # both coexist in one collection


def test_reset_then_reindex_repopulates(tmp_path):
    """Regression: reset() must clear the incremental hash manifest too. Otherwise the next index() sees
    every symbol as 'unchanged' (stale manifest), skips embedding, and leaves an EMPTY collection —
    recall returns nothing. reset()+index() must rebuild fully."""
    db = str(tmp_path / "db")
    mem = CodeMemory(MINI, db_path=db, service="payments")
    mem.index()
    assert mem.recall("reconcile the ledger entry after an account migration", service="payments")["symbols"]

    mem.reset()
    assert not os.path.exists(os.path.join(db, HASH_FILENAME)), "reset() left the hash manifest behind"

    report = CodeMemory(MINI, db_path=db, service="payments").index()
    assert report["indexed"] >= 4, f"reindex after reset embedded nothing (manifest leak): {report}"
    again = CodeMemory(MINI, db_path=db, service="payments").recall(
        "reconcile the ledger entry after an account migration", service="payments")
    assert again["symbols"], "recall empty after reset+reindex — the manifest leak regressed"
