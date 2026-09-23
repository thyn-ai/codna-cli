"""CLI surface for repo intelligence: `codna impact` + `codna memory export`.

Parser wiring and command behavior are tested offline; the telys-backed export path skips cleanly
where the kernel is unavailable (offline CI lane), same posture as test_memory_incremental.py.
"""
from __future__ import annotations

import json
import os

import pytest

from codna import cli


def _parse(argv):
    return cli.build_parser().parse_args(argv)


# ── parser wiring ────────────────────────────────────────────────────────────────────────────────


def test_impact_parser_defaults():
    a = _parse(["impact"])
    assert a.base == "origin/main" and a.head == "HEAD" and a.changed is None
    assert a.as_json is False and a.repo is None
    assert a.shared_core_fraction == pytest.approx(0.34)


def test_impact_parser_flags():
    a = _parse(["impact", "--repo", "/r", "--changed", "a.py", "b.ts", "--json",
                "--shared-core-fraction", "0.5"])
    assert a.repo == "/r" and a.changed == ["a.py", "b.ts"]
    assert a.as_json is True and a.shared_core_fraction == pytest.approx(0.5)


def test_memory_export_parser():
    a = _parse(["memory", "export", "/tmp/art"])
    assert a.path == "/tmp/art" and a.mode == "int8" and a.repo is None
    b = _parse(["memory", "export", "/tmp/art", "--mode", "pq", "--repo", "/r"])
    assert b.mode == "pq" and b.repo == "/r"


def test_memory_export_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        _parse(["memory", "export", "/tmp/art", "--mode", "f32"])


# ── codna impact behavior (offline: tmp repos, --changed to skip git) ───────────────────────────


def _write_repo(root) -> None:
    os.makedirs(os.path.join(root, "src", "pkg"), exist_ok=True)
    os.makedirs(os.path.join(root, "tests"), exist_ok=True)
    with open(os.path.join(root, "src", "pkg", "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(root, "src", "pkg", "core.py"), "w") as f:
        f.write("def core():\n    return 1\n")
    with open(os.path.join(root, "tests", "test_core.py"), "w") as f:
        f.write("def test_core():\n    assert True\n")
    with open(os.path.join(root, "tests", "test_other.py"), "w") as f:
        f.write("def test_other():\n    assert True\n")


class _ImpactArgs:
    def __init__(self, repo, changed):
        self.repo = repo
        self.changed = changed
        self.base = "origin/main"
        self.head = "HEAD"
        self.shared_core_fraction = 0.34
        self.as_json = False


def test_impact_python_change_selects_importing_test(tmp_path, capsys):
    repo = str(tmp_path / "repo")
    _write_repo(repo)
    # filler tests keep the affected share (1/5) under the 0.34 shared-core threshold
    for i in range(3):
        with open(os.path.join(repo, "tests", f"test_fill_{i}.py"), "w") as f:
            f.write(f"def test_fill_{i}():\n    assert True\n")
    with open(os.path.join(repo, "tests", "test_core.py"), "w") as f:
        f.write("from src.pkg.core import core\n\ndef test_core():\n    assert core() == 1\n")

    from codna.impact_cli import cmd_impact
    rc = cmd_impact(_ImpactArgs(repo, ["src/pkg/core.py"]))

    assert rc == 0
    out = capsys.readouterr().out.split()
    assert out == ["tests/test_core.py"]          # only the importing test, not test_other


def _stub_language_registry(monkeypatch, lang_by_ext):
    """Deterministic language classification regardless of which tree-sitter grammars happen to be
    installed in the test environment (the offline CI lane has none -> the real registry would be
    Python-only and the disjoint fast path would never fire)."""
    from codna import codeunits as cu

    class _U:
        symbol_type = "function"

    monkeypatch.setattr(cu, "language_for", lambda rel: lang_by_ext.get(os.path.splitext(rel)[1]))
    monkeypatch.setattr(cu, "extract_file", lambda repo_id, rel, source: [_U()])


def test_impact_non_python_only_diff_selects_nothing(tmp_path, capsys, monkeypatch):
    repo = str(tmp_path / "repo")
    _write_repo(repo)
    with open(os.path.join(repo, "web.ts"), "w") as f:
        f.write("export const x = 1;\n")
    _stub_language_registry(monkeypatch, {".ts": "typescript"})

    from codna.impact_cli import cmd_impact
    rc = cmd_impact(_ImpactArgs(repo, ["web.ts"]))

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""             # subset with zero tests
    assert "non-Python languages" in captured.err


def test_impact_broad_change_prints_all(tmp_path, capsys):
    repo = str(tmp_path / "repo")
    _write_repo(repo)

    from codna.impact_cli import cmd_impact
    rc = cmd_impact(_ImpactArgs(repo, ["pyproject.toml"]))

    assert rc == 0
    assert capsys.readouterr().out.strip() == "ALL"


def test_impact_json_contract(tmp_path, capsys, monkeypatch):
    repo = str(tmp_path / "repo")
    _write_repo(repo)
    _stub_language_registry(monkeypatch, {".ts": "typescript"})

    from codna.impact_cli import cmd_impact
    args = _ImpactArgs(repo, ["web.ts"])
    args.as_json = True
    with open(os.path.join(repo, "web.ts"), "w") as f:
        f.write("export const x = 1;\n")
    assert cmd_impact(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"mode", "tests", "reason"}
    assert payload["mode"] == "subset" and payload["tests"] == []


# ── codna memory export ──────────────────────────────────────────────────────────────────────────


def test_export_without_compact_seam_raises(monkeypatch):
    """Older telys (no export_compact on the collection) must fail with guidance, not write junk."""
    from codna.memory import CodeMemory, CodeMemoryError

    class _NoSeam:
        def stats(self):
            return {"external_ids": 3}

    mem = CodeMemory.__new__(CodeMemory)
    mem.is_empty = lambda: False
    mem.ensure_fresh = lambda: None
    mem._collection = lambda: _NoSeam()

    with pytest.raises(CodeMemoryError, match="telys#99"):
        mem.export_serve_artifact("/tmp/never-written")
    assert not os.path.exists("/tmp/never-written")


def test_export_end_to_end(tmp_path, monkeypatch):
    """Needs the Telys kernel + the compact seam; skips on older stacks (offline CI lane)."""
    monkeypatch.setenv("CODNA_MEMORY_EMBED", "local")
    repo = str(tmp_path / "repo")
    os.makedirs(repo, exist_ok=True)
    for i in range(3):
        with open(os.path.join(repo, f"m{i}.py"), "w") as f:
            f.write("\n".join(f"def func_{i}_{j}(x):\n    '''thing {i} {j}'''\n    return x + {j}"
                              for j in range(4)))
    try:
        from codna.memory import CodeMemory
        mem = CodeMemory(repo, db_path=str(tmp_path / "db"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"telys/kernel unavailable: {type(exc).__name__}: {exc}")
    if not hasattr(mem._collection(), "export_compact"):
        pytest.skip("resolved telys predates the compact seam (telys#99)")

    info = CodeMemory(repo, db_path=str(tmp_path / "db")).export_serve_artifact(
        str(tmp_path / "artifact"))

    assert info["documents"] == 12
    assert info["mode"] == "int8" and info["size_bytes"] > 0
    assert os.path.isfile(os.path.join(info["artifact"], "manifest.json"))
    assert not os.path.exists(os.path.join(info["artifact"], "ty_base.npy"))  # no f32 slab
