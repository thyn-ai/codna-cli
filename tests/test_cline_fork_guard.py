from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cline_fork_guard.py"
_SPEC = importlib.util.spec_from_file_location("cline_fork_guard", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_GUARD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GUARD)


def _write_vendored_cline_attribution(root: Path) -> None:
    vendor = root / "agent-core" / "vendor" / "cline"
    vendor.mkdir(parents=True)
    (vendor / "package.json").write_text("{}", encoding="utf-8")
    (vendor / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (vendor / "NOTICE").write_text("Vendored Cline source\n", encoding="utf-8")


def test_guard_prunes_generated_and_vendored_trees(tmp_path: Path):
    _write_vendored_cline_attribution(tmp_path)
    (tmp_path / "pnpm-lock.yaml").write_text("packages: {}\n", encoding="utf-8")

    ignored_lockfiles = [
        tmp_path / ".codex-tmp" / "pnpm-lock.yaml",
        tmp_path / "node_modules" / "pnpm-lock.yaml",
        tmp_path / "agent-core" / "vendor" / "cline" / "pnpm-lock.yaml",
    ]
    for lockfile in ignored_lockfiles:
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        lockfile.write_text("@cline/core\n", encoding="utf-8")

    assert _GUARD.run_guards(tmp_path) == []


def test_guard_still_rejects_product_lockfile_cline_dependency(tmp_path: Path):
    (tmp_path / "pnpm-lock.yaml").write_text("@cline/sdk\n", encoding="utf-8")

    violations = _GUARD.scan_lockfiles_for_npm_cline(tmp_path)

    assert len(violations) == 1
    assert "our lockfile references npm @cline/core|@cline/sdk" in violations[0]
