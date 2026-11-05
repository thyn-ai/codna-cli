"""Import wiring for a resolved Codna-managed Telys install (split out of memory.py's 1000-line ceiling).

THE DEFECT THIS MODULE EXISTS FOR, MEASURED. ``codna.memory`` resolved a Telys *kernel* from the install root
(``$CODNA_TELYS_INSTALL_ROOT`` or ``build/local-telys`` in a source checkout) and exported ``TELYS_KERNEL`` —
but never made the SIBLING ``site-packages`` directory importable. The process therefore imported whatever
``telys`` happened to be installed system-wide, or none at all, and recall failed with "the Telys runtime is
not installed" while a complete, current install sat on disk. It worked only when the caller set
``PYTHONPATH`` by hand, which is exactly the manual step the product must not require.

``install-manifest.json`` (written by ``scripts/install-local-telys.sh`` and by
``scripts/stage_telys_test_install_root.py`` in CI) already records where those packages live, so the fix is
to honour it: resolve ``site_dir`` from the manifest and make it importable.

Placement matters and is not cosmetic. ``sys.path.insert(0, ...)`` would shadow the vendored
``cli/memengine`` that ships beside ``codna`` with the install's own build; ``sys.path.append`` loses to an
older ``telys`` in the user's site-packages — which is the failure being fixed. The entry therefore goes
IMMEDIATELY BEFORE the first path entry that provides ``telys``: the same position a hand-set
``PYTHONPATH=<site-dir>`` occupies.

Every import in ``codna.memory`` stays lazy; nothing here executes an installed package. Only path arithmetic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

CODNA_INSTALL_ROOT_ENV_NAME = "CODNA_TELYS_INSTALL_ROOT"
INSTALL_MANIFEST_FILENAME = "install-manifest.json"
INSTALL_KERNEL_DIRNAME = "kernel"
INSTALL_SITE_DIRNAME = "site-packages"


def source_checkout_root(package_file: str | Path) -> Path:
    """Root of the Codna source checkout, derived from a file INSIDE the package.

    ``package_file`` is explicit rather than read from ``codna.memory.__file__``: callers and tests point it at
    a synthetic tree, and a module-global read would silently ignore what they passed. This mirrors
    ``codna.telys_runtime.candidate_codna_install_roots``.
    """
    return Path(package_file).resolve().parents[2]


def candidate_install_roots(package_file: str | Path) -> list[tuple[str, Path]]:
    """Install roots to consider, in resolution order: explicit operator config, then the source checkout."""
    roots: list[tuple[str, Path]] = []
    raw = os.environ.get(CODNA_INSTALL_ROOT_ENV_NAME)
    if raw and raw.strip():
        roots.append((f"env:{CODNA_INSTALL_ROOT_ENV_NAME}", Path(raw).expanduser()))
    roots.append(("codna:source-build", source_checkout_root(package_file) / "build" / "local-telys"))
    return roots


def install_manifest_site_dir(install_root: Path) -> Path | None:
    """The ``site_dir`` a Codna-managed install root declares, or ``None`` — never a guess.

    ``None`` when the root is not an install this process can use: no manifest, malformed JSON, not an
    object, no ``site_dir`` key, a non-string/empty value. A declared-but-absent directory is also ``None``:
    the kernel may still be usable (a packaged runtime ships one without Python packages), but there is
    nothing to import, and silently adding a nonexistent path buys nothing.
    """
    manifest_path = install_root / INSTALL_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    declared = manifest.get("site_dir")
    if not isinstance(declared, str) or not declared.strip():
        return None
    # Absolute in every install this repo produces, but resolve a relative one against the install root
    # rather than against the caller's current directory.
    site_dir = Path(declared).expanduser()
    if not site_dir.is_absolute():
        site_dir = install_root / site_dir
    return site_dir.resolve()


def candidate_install_site_dirs(package_file: str | Path) -> list[tuple[str, Path]]:
    """``(source, site_dir)`` pairs for install roots that declare one, in resolution order."""
    resolved: list[tuple[str, Path]] = []
    for source, root in candidate_install_roots(package_file):
        site_dir = install_manifest_site_dir(root)
        if site_dir is not None:
            resolved.append((source, site_dir))
    return resolved


def activation_index(site_dir: Path) -> int:
    """Index at which ``site_dir`` becomes the ``telys`` that gets imported, without shadowing everything.

    Returns the position of the FIRST ``sys.path`` entry that provides a ``telys`` package. Inserting there
    puts the install where a hand-set ``PYTHONPATH=<site-dir>`` would have put it: after the caller's own
    ``codna`` / vendored ``memengine`` entries, ahead of the ambient site-packages. When no entry provides
    ``telys``, appending is equivalent and least disruptive.
    """
    for index, entry in enumerate(sys.path):
        candidate = Path(entry or os.getcwd())
        if (candidate / "telys" / "__init__.py").is_file():
            return index
    return len(sys.path)


def activate_install_site_packages(package_file: str | Path) -> dict:
    """Make the resolved install's ``site_dir`` importable by THIS process. Idempotent and order-stable.

    A directory already ahead of every ``telys`` provider is left alone; one that is present but SHADOWED is
    moved forward to the effective position. Returns what was done, so a caller (or a test) can report it
    rather than assume it.
    """
    activated: list[str] = []
    for source, site_dir in candidate_install_site_dirs(package_file):
        if not site_dir.is_dir():
            continue
        entry = str(site_dir)
        index = activation_index(site_dir)
        present_at = next((i for i, candidate in enumerate(sys.path) if candidate == entry), None)
        # `< index` is not enough: once this entry IS the first `telys` provider, `activation_index` returns
        # its own position, so `present_at == index` means it is already effective and must be left alone.
        # Without this the entry is needlessly removed and re-inserted on every call and reported as
        # "moved ahead of a shadowing telys" when nothing shadows it.
        if present_at is not None and present_at <= index:
            activated.append(f"{source}:{entry} (already effective)")
            continue
        if present_at is not None:
            del sys.path[present_at]
            index = activation_index(site_dir)
        sys.path.insert(index, entry)
        state = "moved ahead of a shadowing telys" if present_at is not None else "inserted"
        activated.append(f"{source}:{entry} ({state} at {index})")
    return {"activated": activated, "sys_path_entry_count": len(sys.path)}
