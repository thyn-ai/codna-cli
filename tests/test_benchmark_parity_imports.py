from __future__ import annotations

import os
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from codna_parity import benchmark_adapters  # noqa: E402


def test_ensure_repo_imports_skips_cross_interpreter_engine_site_packages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine_dir = tmp_path / "decision-engine"
    current_major, current_minor = sys.version_info[:2]
    compatible = (
        engine_dir
        / ".venv"
        / "lib"
        / f"python{current_major}.{current_minor}"
        / "site-packages"
    )
    incompatible = (
        engine_dir
        / ".venv"
        / "lib"
        / f"python{current_major}.{current_minor + 1}"
        / "site-packages"
    )
    compatible.mkdir(parents=True)
    incompatible.mkdir(parents=True)

    monkeypatch.setattr(benchmark_adapters, "DEFAULT_ENGINE_DIR", engine_dir)
    monkeypatch.setattr(sys, "path", ["/active-venv/site-packages"])

    benchmark_adapters.ensure_repo_imports()

    assert str(compatible) in sys.path
    assert str(incompatible) not in sys.path


def test_site_packages_python_version_requires_matching_version_segment() -> None:
    path = Path("/tmp/engine/.venv/lib/python3.12/site-packages")

    assert benchmark_adapters._site_packages_python_version(path) == (3, 12)
    assert benchmark_adapters._site_packages_python_version(Path("/tmp/site-packages")) is None


def test_benchmark_run_uses_output_scoped_runtime_root(monkeypatch, tmp_path: Path) -> None:
    from codna_parity import benchmark_suite

    monkeypatch.delenv("CODNA_RUNTIME_ROOT", raising=False)

    runtime_root = benchmark_suite._ensure_benchmark_runtime_root(tmp_path / "out")

    assert runtime_root == (tmp_path / "out" / "runtime" / "codna").resolve(strict=False)
    assert Path(os.environ["CODNA_RUNTIME_ROOT"]) == runtime_root


def test_benchmark_runtime_root_preserves_explicit_override(monkeypatch, tmp_path: Path) -> None:
    from codna_parity import benchmark_suite

    explicit = tmp_path / "explicit-runtime"
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(explicit))

    runtime_root = benchmark_suite._ensure_benchmark_runtime_root(tmp_path / "out")

    assert runtime_root == explicit.resolve(strict=False)
    assert os.environ["CODNA_RUNTIME_ROOT"] == str(explicit)
