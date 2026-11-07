"""Detect when an upstream sync strips codna's downstream memengine overlay.

Whose bug this is
-----------------
``thyn-ai/telys`` publishes ``packages/memengine-py/memengine`` to codna's ``cli/memengine`` with

    rsync -a --delete --exclude='__pycache__' packages/memengine-py/memengine/ codna/cli/memengine/

(``.github/workflows/sync-memengine-to-codna.yml`` in telys). ``--delete`` makes codna's directory a *mirror* of
the upstream package, which is wrong for this repo: codna carries a downstream overlay upstream has never shipped.
``memengine/_pynative.py`` and ``memengine/_pykernel.py`` exist only here -- the pure-NumPy reference
implementation and the ctypes adapter that answers the kernel's C ABI (AGENTS.md section 24.8), which are also the
subject of ``tests/test_pynative_equivalence.py``. A mirroring rsync deletes them on every sync.

That has happened three times: 77868391, then again on the branch behind PR #697 after 5a13cd08 had restored them.
codna's CI cannot reach the telys repo that owns the workflow, so this module is the only place the invariant can
be enforced. It does not try to repair the tree -- restoring a divergent two-way overlay automatically would mean
merging two codebases on a guess. It makes the loss LOUD and PRECISE, at commit time, instead of surfacing as a
collection error or, worse, a released wheel with no fallback.

The record it checks against is ``_codna_overlay.json``, next to this file.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import sys

MANIFEST_NAME = "_codna_overlay.json"
MANIFEST_SCHEMA_VERSION = 1

_MEMENGINE_DIR = Path(__file__).resolve().parent


class OverlayError(RuntimeError):
    """The overlay is incomplete or its manifest is unusable. Carries structured context for CI logs."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class OverlayEntry:
    """One codna-only file (``symbols`` empty) or one shared module carrying downstream ``symbols``."""

    path: str
    role: str | None = None
    symbols: tuple[str, ...] = ()


def _require(condition: bool, code: str, message: str, **details: object) -> None:
    if not condition:
        raise OverlayError(code, message, **details)


def require_confined_entry(name: str, base: Path, manifest_path: Path) -> None:
    """Reject a 'preserve' path that is absolute, contains '..', or resolves outside ``base``.

    The manifest is a repo file, but every consumer of it feeds the names to ``read_bytes`` and, at publish
    time, writes those bytes into the distributed wheel. An entry of ``../../etc/passwd`` or ``/etc/hosts``
    would pass a mere non-empty-string check, so the whole tree would ship the contents of a file outside the
    package. Entries are therefore confined to the package, and the error names the offending entry.
    """
    if "\\" in name:
        # A literal filename character on POSIX, a separator on Windows: one manifest cannot mean the same
        # file on both, and this tree ships to both.
        raise OverlayError(
            "manifest_path_unsafe",
            f"{manifest_path}: 'preserve' entry {name!r} uses a backslash; entries are relative POSIX paths",
            path=str(manifest_path),
            entry=name,
        )
    relative = PurePosixPath(name)
    if relative.is_absolute():
        raise OverlayError(
            "manifest_path_unsafe",
            f"{manifest_path}: 'preserve' entry {name!r} is absolute; entries must be relative to {base}",
            path=str(manifest_path),
            entry=name,
        )
    if any(part in ("..", "") for part in name.split("/")):
        raise OverlayError(
            "manifest_path_unsafe",
            f"{manifest_path}: 'preserve' entry {name!r} escapes the package with '..'",
            path=str(manifest_path),
            entry=name,
        )
    resolved = (base / name).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise OverlayError(
            "manifest_path_unsafe",
            f"{manifest_path}: 'preserve' entry {name!r} resolves outside the package ({resolved})",
            path=str(manifest_path),
            entry=name,
        )


def load_manifest(memengine_dir: Path | None = None) -> dict:
    """Read and validate ``_codna_overlay.json``. An unknown schema version is rejected, never coerced."""
    base = _MEMENGINE_DIR if memengine_dir is None else Path(memengine_dir)
    path = base / MANIFEST_NAME
    _require(path.is_file(), "manifest_missing", f"overlay manifest not found at {path}", path=str(path))
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # a corrupt manifest must not read as "nothing to preserve"
        raise OverlayError("manifest_invalid", f"{path} is not valid JSON", path=str(path), error=str(exc)) from exc
    _require(isinstance(manifest, dict), "manifest_invalid", f"{path} must contain a JSON object", path=str(path))

    version = manifest.get("schema_version")
    _require(
        version == MANIFEST_SCHEMA_VERSION,
        "manifest_schema_unsupported",
        f"{path} declares schema_version={version!r}; this tool understands {MANIFEST_SCHEMA_VERSION}",
        path=str(path),
        schema_version=version,
    )

    preserved = manifest.get("preserve")
    _require(
        isinstance(preserved, list) and bool(preserved),
        "manifest_invalid",
        f"{path} has no 'preserve' entries — an overlay that preserves nothing is a typo, not a policy",
        path=str(path),
    )
    for entry in preserved:
        _require(
            isinstance(entry, dict) and isinstance(entry.get("path"), str) and bool(entry["path"]),
            "manifest_invalid",
            f"{path}: every 'preserve' entry needs a non-empty 'path'",
            path=str(path),
        )
        require_confined_entry(entry["path"], base, path)

    overlaid = manifest.get("overlaid_modules", [])
    _require(isinstance(overlaid, list), "manifest_invalid", f"{path}: 'overlaid_modules' must be a list")
    for entry in overlaid:
        _require(
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("symbols"), list)
            and bool(entry["symbols"]),
            "manifest_invalid",
            f"{path}: every 'overlaid_modules' entry needs a 'path' and a non-empty 'symbols' list",
            path=str(path),
        )
    return manifest


def preserved_paths(manifest: dict) -> tuple[OverlayEntry, ...]:
    return tuple(OverlayEntry(path=e["path"], role=e.get("role")) for e in manifest["preserve"])


def overlaid_modules(manifest: dict) -> tuple[OverlayEntry, ...]:
    return tuple(
        OverlayEntry(path=e["path"], symbols=tuple(e["symbols"])) for e in manifest.get("overlaid_modules", [])
    )


def missing_preserved(memengine_dir: Path, manifest: dict) -> tuple[str, ...]:
    """Codna-only files the sync deleted. Unconditional: upstream owns none of these paths."""
    base = Path(memengine_dir)
    return tuple(e.path for e in preserved_paths(manifest) if not (base / e.path).is_file())


def defined_names(source: str) -> frozenset[str]:
    """Every name ``source`` DEFINES at module level or in a class body (functions, classes, assignments).

    Parsed, not searched: a synced module whose docstring or comment merely *mentions* ``backend_kind`` must not
    pass for one that defines it, and a prose-only mention is exactly what a mirrored upstream file carries.
    Class bodies are included because the compact-export capability is declared as class attributes on the runtimes.

    Function bodies are NOT descended into. A name bound only inside a function — a local, a nested ``def``, a
    comprehension target — is not something a consumer of the module can import, so counting it would be the same
    false pass as a prose mention: upstream deleting ``backend_kind`` while leaving a local variable of that name
    must still be reported. A class defined inside a function is skipped with it, because such a class is reachable
    only through that function's scope.
    """
    names: set[str] = set()

    def collect(body: list[ast.stmt]) -> None:
        """Record the definitions in one module or class body, descending into classes but not functions."""
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
                collect(node.body)
            elif isinstance(node, ast.Assign):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            else:
                # try/if/with/for blocks at module or class level DEFINE their names there too; recurse so a
                # container-dependent definition is still recognised, while functions stay opaque.
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(node, field, None)
                    if isinstance(nested, list):
                        collect(nested)
                for handler in getattr(node, "handlers", ()) or ():
                    collect(handler.body)
                for case in getattr(node, "cases", ()) or ():
                    collect(case.body)

    collect(ast.parse(source).body)
    return frozenset(names)


def missing_symbols(memengine_dir: Path, manifest: dict) -> tuple[str, ...]:
    """Downstream symbols a synced module no longer defines, as ``"module.py:symbol"``."""
    base = Path(memengine_dir)
    absent: list[str] = []
    for module in overlaid_modules(manifest):
        target = base / module.path
        if not target.is_file():
            # The whole module is gone; report every symbol it owed so one breakage cannot hide a second.
            absent.extend(f"{module.path}:{symbol}" for symbol in module.symbols)
            continue
        defined = defined_names(target.read_text(encoding="utf-8"))
        absent.extend(f"{module.path}:{symbol}" for symbol in module.symbols if symbol not in defined)
    return tuple(absent)


def audit(memengine_dir: Path | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(missing_files, missing_symbols)`` for the vendored tree at ``memengine_dir`` (default: this package)."""
    base = _MEMENGINE_DIR if memengine_dir is None else Path(memengine_dir)
    manifest = load_manifest(base)
    return missing_preserved(base, manifest), missing_symbols(base, manifest)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    unknown = [a for a in args if a not in ("--check", "--report")]
    if unknown:
        raise OverlayError("usage", f"unknown argument {unknown[0]!r}; use --check or --report")
    missing, stripped = audit()
    if "--check" in args:
        if missing or stripped:
            raise OverlayError(
                "overlay_incomplete",
                "the vendored memengine lost codna's downstream overlay (an upstream sync mirrored a package "
                "that does not contain it); restore the files below from main before merging",
                missing=list(missing),
                missing_symbols=list(stripped),
            )
        print("overlay intact")
        return 0
    print(json.dumps({"missing_files": list(missing), "missing_symbols": list(stripped)}, indent=2))
    return 0 if not (missing or stripped) else 1


if __name__ == "__main__":  # pragma: no cover - the CLI body is covered through main() in tests
    try:
        sys.exit(main())
    except OverlayError as error:
        print(json.dumps({"error": error.code, "message": error.message, **error.details}), file=sys.stderr)
        sys.exit(2)
