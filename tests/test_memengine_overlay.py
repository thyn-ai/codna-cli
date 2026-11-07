"""The vendored memengine must keep codna's downstream overlay across an upstream sync.

Background: thyn-ai/telys mirrors ``packages/memengine-py/memengine`` into ``cli/memengine`` with
``rsync -a --delete``, so this directory is treated as a copy of a package that has NEVER contained
``_pynative.py`` or ``_pykernel.py``. Those two files are codna-only: the pure-NumPy reference
implementation and the ctypes adapter that answers the kernel's C ABI (AGENTS.md section 24.8). The
mirroring sync deleted them in 77868391, in the branch behind PR #697 (c7a64c05) after 5a13cd08 had
restored them, and would do so again on the next ``telys`` main push.

These tests are the tripwire. They are structural on purpose: the behavioural contract (that the fallback
actually returns the same rankings as the kernel) lives in ``test_pynative_equivalence.py`` and
``test_pynative_kernel_absent.py``, and these tests must not be able to stand in for it.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

# The audit tool lives next to the engine so it can be run from a checkout with PYTHONPATH=cli, exactly as
# `python3 -m memengine._codna_overlay --check`. Importing it through the package keeps one implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memengine import _codna_overlay as overlay  # noqa: E402

MEMENGINE_DIR = Path(__file__).resolve().parents[1] / "memengine"


def _write_package(root: Path, files: dict[str, str]) -> Path:
    """Materialise a fake memengine package; return it."""
    package = root / "memengine"
    package.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text(source, encoding="utf-8")
    return package


def _manifest(root: Path, *, preserve=("_pynative.py", "_pykernel.py"),
              modules=(("runtime.py", ["supports_compact_export"]),)) -> None:
    (root / "memengine" / overlay.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preserve": [{"path": name, "role": "reference"} for name in preserve],
                "overlaid_modules": [{"path": path, "symbols": list(syms)} for path, syms in modules],
            }
        ),
        encoding="utf-8",
    )


# ── the real tree ─────────────────────────────────────────────────────────────────────────────────
def test_the_repository_tree_still_carries_the_downstream_overlay() -> None:
    """The regression itself.

    Red before the restore: PR #697's tree mirrored an upstream package with no ``_pynative.py``, and the
    failure surfaced as a collection error in an unrelated test file rather than as this clear message.
    """
    missing_files, missing_symbols = overlay.audit(MEMENGINE_DIR)
    assert not missing_files, (
        "an upstream sync deleted codna-only memengine files: "
        f"{missing_files}. They are not in thyn-ai/telys and the sync's `rsync --delete` mirrors that absence. "
        "Restore them from main (`git show main:cli/memengine/<file> > cli/memengine/<file>`)."
    )
    assert not missing_symbols, (
        "an upstream sync reverted downstream code codna depends on: "
        f"{missing_symbols}. Restore those modules from main in the same commit."
    )


def test_the_manifest_lists_the_files_that_are_actually_codna_only() -> None:
    """The manifest is a claim about upstream; a file that upstream DOES ship must not be listed as ours."""
    manifest = overlay.load_manifest(MEMENGINE_DIR)
    names = {entry.path for entry in overlay.preserved_paths(manifest)}
    assert names == {"_pynative.py", "_pykernel.py"}, (
        "the codna-only set drifted from what the vendored tree and tests actually depend on: "
        f"{sorted(names)}"
    )


@pytest.mark.parametrize("name", ["_pynative.py", "_pykernel.py"])
def test_preserved_files_are_tracked_by_git(name: str) -> None:
    """An untracked fallback is invisible to a fresh clone, to the wheel, and to this guard's own premise."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", f"cli/memengine/{name}"],
        cwd=MEMENGINE_DIR.parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"cli/memengine/{name} is not tracked by git: {result.stderr.strip()}"


# ── the audit detects each way a sync can strip the overlay ──────────────────────────────────────
def test_audit_flags_a_preserved_file_the_sync_deleted(tmp_path: Path) -> None:
    _write_package(tmp_path, {"runtime.py": "supports_compact_export = True\n"})
    _manifest(tmp_path)

    missing_files, missing_symbols = overlay.audit(tmp_path / "memengine")

    assert missing_files == ("_pynative.py", "_pykernel.py")
    assert missing_symbols == ()


def test_audit_flags_a_module_whose_downstream_symbol_a_sync_reverted(tmp_path: Path) -> None:
    """The PR did both: it deleted the files AND reverted the modules that consume them."""
    _write_package(tmp_path, {"_pynative.py": "x = 1\n", "_pykernel.py": "y = 1\n", "runtime.py": "OTHER = 1\n"})
    _manifest(tmp_path)

    missing_files, missing_symbols = overlay.audit(tmp_path / "memengine")

    assert missing_files == ()
    assert missing_symbols == ("runtime.py:supports_compact_export",)


def test_a_prose_mention_does_not_satisfy_the_symbol_check(tmp_path: Path) -> None:
    """A mirrored upstream file can still *mention* a symbol in prose while no longer defining it.

    That is precisely the shape of the breakage here: upstream's `mojo_backend` docstring talks about the
    kernel while `backend_kind` is gone. A substring search would call this intact.
    """
    _write_package(
        tmp_path,
        {
            "_pynative.py": "x = 1\n",
            "_pykernel.py": "y = 1\n",
            "runtime.py": '"""supports_compact_export is documented here but not defined."""\n',
        },
    )
    _manifest(tmp_path)

    _missing_files, missing_symbols = overlay.audit(tmp_path / "memengine")

    assert missing_symbols == ("runtime.py:supports_compact_export",)


def test_a_symbol_defined_only_inside_a_class_body_counts(tmp_path: Path) -> None:
    """Capabilities are class attributes on the runtime classes, not module-level names."""
    _write_package(
        tmp_path,
        {
            "_pynative.py": "x = 1\n",
            "_pykernel.py": "y = 1\n",
            "runtime.py": "class LocalRuntime:\n    supports_compact_export = True\n",
        },
    )
    _manifest(tmp_path)

    assert overlay.audit(tmp_path / "memengine") == ((), ())


def test_a_whole_module_deleted_reports_every_symbol_it_owed(tmp_path: Path) -> None:
    """One breakage must not mask another: report all owed symbols, not the first."""
    _write_package(tmp_path, {"_pynative.py": "x = 1\n", "_pykernel.py": "y = 1\n"})
    _manifest(tmp_path, modules=(("runtime.py", ["supports_compact_export", "supports_compact_open"]),))

    _missing_files, missing_symbols = overlay.audit(tmp_path / "memengine")

    assert missing_symbols == ("runtime.py:supports_compact_export", "runtime.py:supports_compact_open")


# ── the manifest itself must fail loudly when it is wrong ────────────────────────────────────────
def test_a_missing_manifest_is_an_error_not_an_empty_policy(tmp_path: Path) -> None:
    _write_package(tmp_path, {"_pynative.py": "x = 1\n"})

    with pytest.raises(overlay.OverlayError) as excinfo:
        overlay.audit(tmp_path / "memengine")

    assert excinfo.value.code == "manifest_missing"


def test_a_manifest_preserving_nothing_is_rejected(tmp_path: Path) -> None:
    _write_package(tmp_path, {"runtime.py": "x = 1\n"})
    (tmp_path / "memengine" / overlay.MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "preserve": []}), encoding="utf-8"
    )

    with pytest.raises(overlay.OverlayError) as excinfo:
        overlay.audit(tmp_path / "memengine")

    assert excinfo.value.code == "manifest_invalid"


def test_an_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    """A future manifest must be understood before it is trusted — not silently read as this schema."""
    _write_package(tmp_path, {"_pynative.py": "x = 1\n"})
    (tmp_path / "memengine" / overlay.MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 2, "preserve": [{"path": "_pynative.py"}]}), encoding="utf-8"
    )

    with pytest.raises(overlay.OverlayError) as excinfo:
        overlay.audit(tmp_path / "memengine")

    assert excinfo.value.code == "manifest_schema_unsupported"


def test_corrupt_manifest_json_raises_rather_than_reporting_intact(tmp_path: Path) -> None:
    _write_package(tmp_path, {"_pynative.py": "x = 1\n"})
    (tmp_path / "memengine" / overlay.MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(overlay.OverlayError) as excinfo:
        overlay.audit(tmp_path / "memengine")

    assert excinfo.value.code == "manifest_invalid"


# ── the CLI contract CI runs ────────────────────────────────────────────────────────────────────
def test_check_exits_zero_on_the_real_tree(capsys) -> None:
    assert overlay.main(["--check"]) == 0
    assert "overlay intact" in capsys.readouterr().out


def test_check_exits_nonzero_and_names_the_losses(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_package(tmp_path, {"runtime.py": "OTHER = 1\n"})
    _manifest(tmp_path)
    monkeypatch.setattr(overlay, "_MEMENGINE_DIR", tmp_path / "memengine")

    with pytest.raises(overlay.OverlayError) as excinfo:
        overlay.main(["--check"])

    assert excinfo.value.code == "overlay_incomplete"
    assert excinfo.value.details["missing"] == ["_pynative.py", "_pykernel.py"]
    assert excinfo.value.details["missing_symbols"] == ["runtime.py:supports_compact_export"]


def test_report_lists_the_losses_without_claiming_success(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_package(tmp_path, {"runtime.py": "OTHER = 1\n"})
    _manifest(tmp_path)
    monkeypatch.setattr(overlay, "_MEMENGINE_DIR", tmp_path / "memengine")

    assert overlay.main(["--report"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing_files"] == ["_pynative.py", "_pykernel.py"]


def test_an_unknown_argument_is_refused(tmp_path: Path) -> None:
    with pytest.raises(overlay.OverlayError) as excinfo:
        overlay.main(["--fix"])

    assert excinfo.value.code == "usage"


# ── a definition is module- or class-level, never a local ────────────────────────────────────────
@pytest.mark.parametrize(
    "source",
    [
        "def helper():\n    supports_compact_export = True\n",
        "def helper():\n    def supports_compact_export(): pass\n",
        "def helper():\n    if True:\n        supports_compact_export = True\n",
        "def helper():\n    supports_compact_export: bool = True\n",
        "def helper():\n    class Inner:\n        supports_compact_export = True\n",
        "class Holder:\n    def helper(self):\n        supports_compact_export = True\n",
    ],
)
def test_a_name_bound_only_inside_a_function_is_not_a_definition(source: str) -> None:
    """A local is the same class of false pass as a prose mention: a consumer cannot import it.

    Upstream can delete ``supports_compact_export`` while leaving a local of that name in some method, and
    an ``ast.walk`` that descends into function bodies would call the module intact.
    """
    assert "supports_compact_export" not in overlay.defined_names(source)


@pytest.mark.parametrize(
    "source",
    [
        "supports_compact_export = True\n",
        "supports_compact_export: bool = True\n",
        "class LocalRuntime:\n    supports_compact_export = True\n",
        "class LocalRuntime:\n    supports_compact_export = True\n    def open_compact(self): ...\n",
        "try:\n    supports_compact_export = True\nexcept ImportError:\n    pass\n",
        "if True:\n    supports_compact_export = True\n",
    ],
)
def test_a_module_or_class_level_definition_still_counts(source: str) -> None:
    assert "supports_compact_export" in overlay.defined_names(source)


def test_a_symbol_defined_only_inside_a_function_does_not_satisfy_the_audit(tmp_path: Path) -> None:
    """End to end: the audit reports the symbol upstream no longer defines at module or class level."""
    _write_package(
        tmp_path,
        {
            "_pynative.py": "x = 1\n",
            "_pykernel.py": "y = 1\n",
            "runtime.py": "def _probe():\n    supports_compact_export = True\n",
        },
    )
    _manifest(tmp_path)

    assert overlay.audit(tmp_path / "memengine") == ((), ("runtime.py:supports_compact_export",))


# ── 'preserve' entries cannot point outside the package ─────────────────────────────────────────
@pytest.mark.parametrize(
    "entry",
    ["../../etc/passwd", "../escape.py", "/etc/hosts", "sub/../../escape.py", "..\\escape.py"],
)
def test_a_preserve_entry_that_escapes_the_package_is_rejected(tmp_path: Path, entry: str) -> None:
    """The manifest is a repo file consumed by a publish path: an escape would put a host file in the wheel."""
    _write_package(tmp_path, {"_pynative.py": "x = 1\n"})
    (tmp_path / "memengine" / overlay.MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "preserve": [{"path": entry}]}), encoding="utf-8"
    )

    with pytest.raises(overlay.OverlayError) as excinfo:
        overlay.audit(tmp_path / "memengine")

    assert excinfo.value.code == "manifest_path_unsafe"
    assert excinfo.value.details["entry"] == entry


def test_a_nested_preserve_entry_inside_the_package_is_accepted(tmp_path: Path) -> None:
    _write_package(tmp_path, {"_pynative.py": "x = 1\n", "kernels/fallback.py": "z = 1\n"})
    (tmp_path / "memengine" / overlay.MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "preserve": [{"path": "kernels/fallback.py"}]}), encoding="utf-8"
    )

    assert overlay.audit(tmp_path / "memengine") == ((), ())
