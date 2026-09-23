from __future__ import annotations

from pathlib import Path
import shutil
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codna._wheel_guard import package_runtime_dir, requires_platform_wheel

AGENT_CORE_EXCLUDED_NAMES = {
    ".codna",
    ".codna-memory",
    ".env",
    ".git",
    ".keys.txt",
    ".pytest_cache",
    ".turbo",
    "__pycache__",
    "build",
    "dist",
    "keys.txt",
    "launcher.state",
    "local-stack.json",
    "node_modules",
}
AGENT_CORE_EXCLUDED_PREFIXES = (".env.",)
AGENT_CORE_EXCLUDED_SUFFIXES = (".jwt", ".key", ".log", ".pem")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_agent_core_dir() -> Path:
    candidates = (
        _repo_root() / "agent-core",
        Path(__file__).resolve().parent / "codna" / "agent-core",
    )
    for candidate in candidates:
        if (candidate / "run-server.mjs").is_file() and (candidate / "run-server.bundle.mjs").is_file():
            return candidate
    raise RuntimeError("agent-core runtime source is missing run-server.mjs or run-server.bundle.mjs")


def _is_excluded_agent_core_name(name: str) -> bool:
    return (
        name in AGENT_CORE_EXCLUDED_NAMES
        or name.startswith(AGENT_CORE_EXCLUDED_PREFIXES)
        or name.endswith(AGENT_CORE_EXCLUDED_SUFFIXES)
    )


def _copy_agent_core_runtime(destination: Path) -> None:
    source = _source_agent_core_dir()
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if _is_excluded_agent_core_name(name)}

    shutil.copytree(source, destination, ignore=ignore)


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError as exc:  # pragma: no cover - build-system.requires installs wheel.
    raise RuntimeError("building codna wheels requires the 'wheel' build dependency") from exc


class build_py(_build_py):
    """Bundle the local agent-core sidecar runtime inside the codna wheel."""

    def run(self) -> None:
        super().run()
        _copy_agent_core_runtime(Path(self.build_lib) / "codna" / "agent-core")


class sdist(_sdist):
    """Include agent-core in source distributions built from the cli package."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        _copy_agent_core_runtime(Path(base_dir) / "codna" / "agent-core")


class bdist_wheel(_bdist_wheel):
    """Mark release wheels platform-specific when they include a native Telys kernel."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = not requires_platform_wheel(package_runtime_dir())

    def get_tag(self):  # type: ignore[no-untyped-def]
        if not requires_platform_wheel(package_runtime_dir()):
            return self.python_tag, "none", "any"
        python_tag, abi_tag, platform_tag = super().get_tag()
        if platform_tag == "any":
            raise RuntimeError(
                "codna package-runtime contains a native Telys kernel but wheel tag is universal"
            )
        return self.python_tag or python_tag, "none", platform_tag


setup(cmdclass={"build_py": build_py, "sdist": sdist, "bdist_wheel": bdist_wheel})
