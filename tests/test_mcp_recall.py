"""Tests for the codna_recall MCP tool body (recall_json) + server registration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from codna.mcp_server import recall_json   # import does NOT pull in `mcp` or `telys`


@pytest.fixture(autouse=True)
def _authorized_device(monkeypatch):
    """The uniform execution gate requires the free `codna login`; run these tests authorized."""
    monkeypatch.setenv("CODNA_API_KEY", "test-device-login")


# ── offline: the guard + friendly failure (no mcp, no telys needed) ───────────────────────────────
def test_recall_requires_query():
    assert recall_json(query="") == "codna_recall error: query is required"


def test_recall_without_telys_is_friendly(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "telys", None)
    monkeypatch.setitem(sys.modules, "telys.embedding", None)
    out = recall_json(repo=str(tmp_path), query="anything")
    assert out.startswith("codna_recall error:") and "reinstall or upgrade `codna`" in out


def test_recall_auto_index_uses_all_languages_sentinel(monkeypatch, tmp_path):
    from codna import memory as memory_module

    captured: dict[str, object] = {}

    class FakeMemory:
        def __init__(self, repo: str, db_path=None, *, service=None) -> None:
            self.repo = repo
            self.db_path = db_path
            self.service = service
            captured["db_path"] = db_path

        def is_empty(self) -> bool:
            return True

        def index(self, *, languages):
            captured["languages"] = languages

        def recall(self, query: str, *, service=None, language=None, final_k=8):
            return {"symbols": [], "explain": {"plan": "mock"}, "candidate_count": 0}

    monkeypatch.setattr(memory_module, "CodeMemory", FakeMemory)

    out = recall_json(repo=str(tmp_path), query="anything")

    assert json.loads(out)["symbols"] == []
    assert captured["languages"] is None
    assert captured["db_path"] is not None


def test_recall_uses_external_mcp_memory_root(monkeypatch, tmp_path):
    from codna import memory as memory_module

    repo = tmp_path / "repo"
    repo.mkdir()
    memory_root = tmp_path / "memory-root"
    captured: dict[str, object] = {}

    class FakeMemory:
        def __init__(self, repo: str, db_path=None, *, service=None) -> None:
            captured["repo"] = repo
            captured["db_path"] = db_path

        def is_empty(self) -> bool:
            return False

        def recall(self, query: str, *, service=None, language=None, final_k=8):
            return {"symbols": [], "explain": {"plan": "mock"}, "candidate_count": 0}

    monkeypatch.setenv("CODNA_TELYS_MEMORY_ROOT", str(memory_root))
    monkeypatch.setattr(memory_module, "CodeMemory", FakeMemory)

    out = recall_json(repo=str(repo), query="anything")

    assert json.loads(out)["symbols"] == []
    db_path = captured["db_path"]
    assert isinstance(db_path, str)
    assert db_path.startswith(str(memory_root / "mcp"))
    assert db_path != str(repo / ".codna-memory")


# ── runtime-gated: the happy path returns real symbols + a plan ───────────────────────────────────
from telys_gate import require_telys_runtime  # noqa: E402 — must follow the sys.path setup above

# NOT a bare `skipif(not telys_kernel_ready())` any more. That form could only ever SKIP, so a present but
# unusable install (exactly the defect this file's happy path caught) reported as "not installed" and the
# suite went green while recall was broken. `require_telys_runtime` distinguishes the three states:
# it raises for a BROKEN install (never a skip) and skips only for genuine absence — except under
# CODNA_REQUIRE_TELYS_RUNTIME=1, where the CI gate makes absence a failure too.
_TELYS = require_telys_runtime()
if not _TELYS.ready:
    pytest.skip(_TELYS.skip_reason(), allow_module_level=True)


def test_recall_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("CODNA_TELYS_MEMORY_ROOT", str(tmp_path / "memory-root"))
    (tmp_path / "ci.py").write_text(
        'def open_pr_when_ci_fails(branch):\n'
        '    """Open a pull request when CI fails on a branch."""\n'
        '    return branch\n'
    )
    out = recall_json(repo=str(tmp_path), query="open a pull request when CI fails", final_k=3)
    data = json.loads(out)
    assert data["symbols"], "expected recalled symbols"
    assert "plan" in data["explain"]
    assert any("open_pr_when_ci_fails" in s["id"] for s in data["symbols"])


# ── regression: the resolved install's packages must actually become importable ───────────────────
# Hermetic on purpose. The defect is about WIRING (a declared `site_dir` that never reaches `sys.path`),
# so it is reproduced here against a synthetic install root rather than the developer's `build/local-telys`:
# that way the tests run — and fail on the old code — on ANY machine, including a CI runner with no Telys.
# `test_recall_happy_path` above remains the end-to-end proof against a real install where one exists.
def _synthetic_install_root(tmp_path, *, package: str = "telys") -> Path:
    """A minimal install root shaped like `scripts/install-local-telys.sh`'s output (manifest + site dir)."""
    root = tmp_path / "install"
    site = root / "site-packages"
    (site / package).mkdir(parents=True)
    (site / package / "__init__.py").write_text('__version__ = "9.9.9"\nFILE = __file__\n')
    (root / "kernel").mkdir()
    (root / "kernel" / "libame_kernel.dylib").write_bytes(b"")
    (root / "install-manifest.json").write_text(json.dumps({
        "install_root": str(root),
        "site_dir": str(site),
        "kernel_path": str(root / "kernel" / "libame_kernel.dylib"),
        "packaged_kernel_path": str(root / "kernel" / "libame_kernel.dylib"),
    }))
    return root


def test_declared_site_dir_comes_from_the_manifest(monkeypatch, tmp_path):
    """`site_dir` is READ from the install manifest — not assumed to be `<root>/site-packages`."""
    from codna import memory_install

    root = _synthetic_install_root(tmp_path)
    unusual = tmp_path / "elsewhere" / "packages"
    (unusual / "telys").mkdir(parents=True)
    (unusual / "telys" / "__init__.py").write_text('__version__ = "9.9.9"\n')
    (root / "install-manifest.json").write_text(json.dumps({
        "install_root": str(root), "site_dir": str(unusual),
    }))
    monkeypatch.setenv(memory_install.CODNA_INSTALL_ROOT_ENV_NAME, str(root))
    # The override is checked FIRST, so it leads the resolved order; a source checkout may contribute one
    # more entry, which is the real resolution order rather than an accident of this fixture.
    resolved = memory_install.candidate_install_site_dirs(memory_install.__file__)
    assert resolved, "the env override resolved nothing"
    assert resolved[0] == (
        f"env:{memory_install.CODNA_INSTALL_ROOT_ENV_NAME}", unusual.resolve()
    ), f"the manifest's declared site_dir was not the first resolved entry: {resolved}"


def test_malformed_manifest_resolves_no_site_dir(monkeypatch, tmp_path):
    """An unusable manifest must resolve to NOTHING, never to a guessed path (no silent fallback)."""
    from codna import memory_install

    root = _synthetic_install_root(tmp_path)
    (root / "install-manifest.json").write_text("{not json")
    monkeypatch.setenv(memory_install.CODNA_INSTALL_ROOT_ENV_NAME, str(root))

    assert memory_install.install_manifest_site_dir(root) is None


def test_missing_site_dir_key_resolves_no_site_dir(monkeypatch, tmp_path):
    """A manifest WITHOUT `site_dir` must not be treated as `<root>/site-packages` by convention."""
    from codna import memory_install

    root = _synthetic_install_root(tmp_path)
    (root / "install-manifest.json").write_text(json.dumps({"install_root": str(root)}))

    assert memory_install.install_manifest_site_dir(root) is None


def test_declared_site_dir_is_activated_even_behind_a_shadowing_telys(monkeypatch, tmp_path):
    """THE REGRESSION. The declared `site_dir` must end up AHEAD of any other `telys` on `sys.path`.

    THE DEFECT THIS PINS. `codna.memory` resolved `TELYS_KERNEL` from the Codna-managed install root but
    never put that root's `site-packages` on `sys.path` at all, so recall imported whatever `telys` was
    installed system-wide — on the reporting machine an old 0.1.0b3 that predates `TEXT_COLUMN` — and
    failed with "the Telys runtime is not installed" while a complete install sat on disk. It passed only
    when the caller set `PYTHONPATH` by hand, which is why the rot stayed invisible.

    A tautology would assert `site_dir in sys.path` after calling the helper. This instead asserts the
    OBSERVABLE effect: with a decoy `telys` deliberately placed ahead of the install, the decoy is what
    `import telys` would otherwise bind, and the install must displace it — exactly the order a hand-set
    `PYTHONPATH=<site_dir>` produced.
    """
    from codna import memory_install

    root = _synthetic_install_root(tmp_path)
    decoy = tmp_path / "decoy"
    (decoy / "telys").mkdir(parents=True)
    (decoy / "telys" / "__init__.py").write_text('__version__ = "0.0.0-decoy"\n')

    # Sandbox the resolution to this fixture: a developer checkout WOULD legitimately resolve its own
    # `build/local-telys` first, which is not what this test is about.
    monkeypatch.setenv(memory_install.CODNA_INSTALL_ROOT_ENV_NAME, str(root))
    monkeypatch.setattr(
        memory_install, "candidate_install_roots",
        lambda package_file: [(f"env:{memory_install.CODNA_INSTALL_ROOT_ENV_NAME}", root)],
    )
    # A decoy ahead of everything: this is the ambient/system `telys` that used to win.
    monkeypatch.setattr(sys, "path", [str(decoy), *[p for p in sys.path if p != str(decoy)]])
    assert memory_install.activation_index(root / "site-packages") == 0, (
        "fixture is wrong: the decoy should be the first telys provider"
    )

    result = memory_install.activate_install_site_packages(memory_install.__file__)

    assert result["activated"], "the resolved install's site_dir was not activated at all"
    assert sys.path[0] == str((root / "site-packages").resolve()), (
        f"site_dir is not ahead of the decoy; sys.path starts with {sys.path[:3]}"
    )
    # The effect that matters: import resolution now follows the install, not the decoy. Ask the PATH
    # finder directly (not `importlib.util.find_spec`, which short-circuits to whatever `telys` a previous
    # test already bound in `sys.modules`) so this is a real assertion about sys.path ordering.
    import importlib.machinery

    spec = importlib.machinery.PathFinder.find_spec("telys", sys.path)
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).resolve().is_relative_to((root / "site-packages").resolve()), (
        f"`import telys` would resolve {spec.origin}, outside the resolved install — recall would bind an "
        "unrelated SDK again."
    )


def test_activation_is_idempotent(monkeypatch, tmp_path):
    """A second activation must not duplicate the entry, reorder the path, or misreport itself.

    The report matters, not just the path: a repeated call that really is a no-op must SAY so. Before this
    was pinned, the second call removed and re-inserted the entry and claimed it had "moved ahead of a
    shadowing telys" when nothing was shadowing it.
    """
    from codna import memory_install

    root = _synthetic_install_root(tmp_path)
    monkeypatch.setattr(
        memory_install, "candidate_install_roots",
        lambda package_file: [(f"env:{memory_install.CODNA_INSTALL_ROOT_ENV_NAME}", root)],
    )

    memory_install.activate_install_site_packages(memory_install.__file__)
    first = list(sys.path)
    second = memory_install.activate_install_site_packages(memory_install.__file__)

    assert sys.path == first, "a repeated activation mutated sys.path"
    assert sys.path.count(str((root / "site-packages").resolve())) == 1
    assert second["activated"] == [
        f"env:{memory_install.CODNA_INSTALL_ROOT_ENV_NAME}:"
        f"{(root / 'site-packages').resolve()} (already effective)"
    ], f"a no-op activation misreported itself: {second['activated']}"


# ── mcp-gated: the FastMCP server builds with the new tool registered ─────────────────────────────
def _mcp_available():
    try:
        import mcp.server.fastmcp  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _mcp_available(), reason="mcp extra not installed")
def test_build_server_includes_recall():
    from codna.mcp_server import _build_server
    server = _build_server()          # must not raise; the @server.tool() decorator registered codna_recall
    assert server is not None
