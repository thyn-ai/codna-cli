from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_cli_wheel_tags.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_cli_wheel_tags", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AGENT_CORE_RUNTIME_NAMES = (
    "codna/agent-core/run-server.mjs",
    "codna/agent-core/run-server.bundle.mjs",
    "codna/agent-core/run-server.stub.mjs",
    "codna/agent-core/package.json",
    "codna/agent-core/vendor/cline/package.json",
    "codna/agent-core/vendor/cline/bun.lock",
    "codna/agent-core/vendor/cline/algenta/run-server.ts",
)


def _write_fake_wheel(
    path: Path,
    *,
    root_is_purelib: bool,
    tag: str,
    names: tuple[str, ...] = (),
    include_agent_core: bool = True,
) -> None:
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        f"Root-Is-Purelib: {str(root_is_purelib).lower()}\n"
        f"Tag: {tag}\n"
    )
    package_names = (*AGENT_CORE_RUNTIME_NAMES, *names) if include_agent_core else names
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("codna-0.1.0.dist-info/WHEEL", wheel_metadata)
        for name in package_names:
            archive.writestr(name, b"payload")


def _sdist_names(*names: str) -> tuple[str, ...]:
    agent_core_names = tuple(f"codna-0.1.0/{name}" for name in AGENT_CORE_RUNTIME_NAMES)
    return (*agent_core_names, *names)


def _agent_core_payload(*, node_modules: bool = False) -> dict[str, object]:
    return {
        "exists": True,
        "run_server": True,
        "runtime_bundle": True,
        "vendored_server": True,
        "vendored_package": True,
        "vendored_lock": True,
        "node_modules": node_modules,
    }


def _package_metadata() -> dict[str, object]:
    return {
        "description_content_type": "text/markdown",
        "readme_marker": True,
        "summary": "Codna package metadata",
        "version": "0.1.0",
    }


def test_smoke_env_disables_pip_keyring_only(monkeypatch):
    tool = _load_script()
    monkeypatch.delenv("CODNA_DISABLE_KEYCHAIN", raising=False)
    monkeypatch.delenv("PIP_KEYRING_PROVIDER", raising=False)

    env = tool.smoke_env({})

    assert "CODNA_DISABLE_KEYCHAIN" not in env
    assert env["PIP_KEYRING_PROVIDER"] == "disabled"


def test_assert_pure_wheel_accepts_universal_wheel(tmp_path: Path):
    tool = _load_script()
    wheel = tmp_path / "codna-0.1.0-py3-none-any.whl"
    _write_fake_wheel(wheel, root_is_purelib=True, tag="py3-none-any")

    tool.assert_pure_wheel(tool.inspect_wheel(wheel))


def test_assert_pure_wheel_rejects_missing_agent_core_bundle(tmp_path: Path):
    tool = _load_script()
    wheel = tmp_path / "codna-0.1.0-py3-none-any.whl"
    _write_fake_wheel(
        wheel,
        root_is_purelib=True,
        tag="py3-none-any",
        names=tuple(name for name in AGENT_CORE_RUNTIME_NAMES if not name.endswith("run-server.bundle.mjs")),
        include_agent_core=False,
    )

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_pure_wheel(tool.inspect_wheel(wheel))

    assert "run-server.bundle.mjs" in str(excinfo.value)


def test_assert_pure_wheel_rejects_runtime_artifacts(tmp_path: Path):
    tool = _load_script()
    wheel = tmp_path / "codna-0.1.0-py3-none-any.whl"
    _write_fake_wheel(
        wheel,
        root_is_purelib=True,
        tag="py3-none-any",
        names=("codna/_telys_runtime/manifest.json",),
    )

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_pure_wheel(tool.inspect_wheel(wheel))

    assert "runtime artifacts" in str(excinfo.value)


def test_copy_cli_source_scrubs_all_staged_runtime_artifacts(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / "source"
    runtime_dir = source / "codna" / "_telys_runtime"
    runtime_dir.mkdir(parents=True)
    (source / "codna" / "__init__.py").write_text("", encoding="utf-8")
    keep = runtime_dir / "README.md"
    keep.write_text("runtime docs\n", encoding="utf-8")
    for name in (
        "libame_kernel.dylib",
        "libAsyncRTMojoBindings.dylib",
        "libty_runtime.so",
        "ame_kernel.dll",
        "manifest.json",
        "oem-license.jwt",
    ):
        (runtime_dir / name).write_bytes(b"staged")

    destination = tmp_path / "copy"
    tool.copy_cli_source(source, destination)

    copied_runtime = destination / "codna" / "_telys_runtime"
    assert (copied_runtime / "README.md").read_text(encoding="utf-8") == "runtime docs\n"
    assert not any(
        path.name != "README.md" for path in copied_runtime.iterdir()
    )


@pytest.mark.parametrize(
    "name",
    [
        "codna/agent-core/.env",
        "codna/agent-core/vendor/cline/node_modules/pkg/package.json",
    ],
)
def test_assert_agent_core_runtime_rejects_local_artifacts(tmp_path: Path, name: str):
    tool = _load_script()
    wheel = tmp_path / "codna-0.1.0-py3-none-any.whl"
    _write_fake_wheel(
        wheel,
        root_is_purelib=True,
        tag="py3-none-any",
        names=(name,),
    )

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_pure_wheel(tool.inspect_wheel(wheel))

    assert "agent-core runtime package includes local-only artifacts" in str(excinfo.value)


def test_assert_native_wheel_rejects_universal_tag(tmp_path: Path):
    tool = _load_script()
    wheel = tmp_path / "codna-0.1.0-py3-none-any.whl"
    _write_fake_wheel(
        wheel,
        root_is_purelib=False,
        tag="py3-none-any",
        names=(
            f"codna/_telys_runtime/{tool.expected_artifact_name()}",
            "codna/_telys_runtime/manifest.json",
            "codna/_telys_runtime/oem-license.jwt",
        ),
    )

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_native_wheel(tool.inspect_wheel(wheel))

    assert "must not be py3-none-any" in str(excinfo.value)


def test_assert_native_wheel_accepts_platform_tag(tmp_path: Path):
    tool = _load_script()
    wheel = tmp_path / "codna-0.1.0-py3-none-manylinux_2_28_x86_64.whl"
    _write_fake_wheel(
        wheel,
        root_is_purelib=False,
        tag="py3-none-manylinux_2_28_x86_64",
        names=(
            f"codna/_telys_runtime/{tool.expected_artifact_name()}",
            "codna/_telys_runtime/manifest.json",
            "codna/_telys_runtime/oem-license.jwt",
        ),
    )

    tool.assert_native_wheel(tool.inspect_wheel(wheel))


def test_assert_sdist_excludes_local_artifacts_accepts_clean_sdist(tmp_path: Path):
    tool = _load_script()
    sdist = tool.SdistInspection(
        path=tmp_path / "codna-0.1.0.tar.gz",
        names=_sdist_names(
            "codna-0.1.0/pyproject.toml",
            "codna-0.1.0/codna/cli.py",
            "codna-0.1.0/codna/_telys_runtime/README.md",
        ),
    )

    tool.assert_sdist_excludes_local_artifacts(sdist)


@pytest.mark.parametrize(
    "name",
    [
        "codna-0.1.0/codna/_telys_runtime/libame_kernel.dylib",
        "codna-0.1.0/codna/_telys_runtime/libame_kernel.so",
        "codna-0.1.0/codna/_telys_runtime/libame_kernel.dll",
        "codna-0.1.0/codna/_telys_runtime/libAsyncRTMojoBindings.dylib",
        "codna-0.1.0/codna/_telys_runtime/libty_runtime.so",
        "codna-0.1.0/codna/_telys_runtime/ame_kernel.dll",
        "codna-0.1.0/codna/_telys_runtime/manifest.json",
        "codna-0.1.0/codna/_telys_runtime/oem-license.jwt",
        "codna-0.1.0/keys.txt",
        "codna-0.1.0/.env",
        "codna-0.1.0/.env.local",
        "codna-0.1.0/private.pem",
        "codna-0.1.0/private.key",
        "codna-0.1.0/local-stack.json",
        "codna-0.1.0/runtime.log",
        "codna-0.1.0/.codna-memory/index.json",
    ],
)
def test_assert_sdist_excludes_local_artifacts_rejects_forbidden_names(tmp_path: Path, name: str):
    tool = _load_script()
    sdist = tool.SdistInspection(path=tmp_path / "codna-0.1.0.tar.gz", names=_sdist_names(name))

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_sdist_excludes_local_artifacts(sdist)

    assert "local-only artifacts" in str(excinfo.value)


def test_assert_install_probe_rejects_missing_entrypoint():
    tool = _load_script()
    payload = {"console_scripts": [], "kernel": {"found": False, "source": "missing"}}

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_install_probe(payload, expect_native_runtime=False)

    assert "entrypoint" in str(excinfo.value)


def test_assert_install_probe_accepts_pure_missing_runtime():
    tool = _load_script()
    payload = {
        "console_scripts": ["codna=codna.cli:main"],
        "kernel": {"found": False, "source": "missing"},
        "metadata": _package_metadata(),
        "version": "0.1.0",
        "agent_core": _agent_core_payload(),
    }

    tool.assert_install_probe(payload, expect_native_runtime=False)


def test_assert_install_probe_rejects_missing_agent_core_status():
    tool = _load_script()
    payload = {
        "console_scripts": ["codna=codna.cli:main"],
        "kernel": {"found": False, "source": "missing"},
        "metadata": _package_metadata(),
        "version": "0.1.0",
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_install_probe(payload, expect_native_runtime=False)

    assert "agent-core package status" in str(excinfo.value)


def test_assert_install_probe_rejects_pure_package_runtime():
    tool = _load_script()
    payload = {
        "console_scripts": ["codna=codna.cli:main"],
        "kernel": {"found": True, "source": "codna:package-runtime"},
        "metadata": _package_metadata(),
        "version": "0.1.0",
        "agent_core": _agent_core_payload(),
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_install_probe(payload, expect_native_runtime=False)

    assert "unexpectedly resolved package runtime" in str(excinfo.value)


def test_assert_install_probe_accepts_native_package_runtime():
    tool = _load_script()
    payload = {
        "console_scripts": ["codna=codna.cli:main"],
        "kernel": {
            "found": True,
            "manifest_path": "/tmp/manifest.json",
            "path": "/tmp/libame_kernel.so",
            "platform": "linux-x86_64",
            "sha256": "0" * 64,
            "size_bytes": 1,
            "source": "codna:package-runtime",
        },
        "metadata": _package_metadata(),
        "version": "0.1.0",
        "agent_core": _agent_core_payload(),
    }

    tool.assert_install_probe(payload, expect_native_runtime=True)


def test_assert_install_probe_rejects_missing_metadata():
    tool = _load_script()
    payload = {
        "console_scripts": ["codna=codna.cli:main"],
        "kernel": {"found": False, "source": "missing"},
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_install_probe(payload, expect_native_runtime=False)

    assert "package metadata" in str(excinfo.value)


def test_assert_install_probe_rejects_metadata_without_readme():
    tool = _load_script()
    metadata = _package_metadata()
    metadata["readme_marker"] = False
    payload = {
        "console_scripts": ["codna=codna.cli:main"],
        "kernel": {"found": False, "source": "missing"},
        "metadata": metadata,
        "version": "0.1.0",
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_install_probe(payload, expect_native_runtime=False)

    assert "README content" in str(excinfo.value)


def test_assert_install_probe_rejects_version_mismatch():
    tool = _load_script()
    metadata = _package_metadata()
    metadata["version"] = "0.1.0"
    payload = {
        "console_scripts": ["codna=codna.cli:main"],
        "kernel": {"found": False, "source": "missing"},
        "metadata": metadata,
        "version": "0.2.0",
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_install_probe(payload, expect_native_runtime=False)

    assert "version does not match" in str(excinfo.value)


def test_assert_entrypoint_help_accepts_codna_usage():
    tool = _load_script()

    tool.assert_entrypoint_help("usage: codna [-h] {doctor,memory}\n")


def test_assert_entrypoint_help_rejects_unrelated_output():
    tool = _load_script()

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_entrypoint_help("hello\n")

    assert "help text" in str(excinfo.value)


def test_assert_module_import_probe_accepts_success():
    tool = _load_script()
    payload = {
        "failures": {},
        "modules": ["cli", "memory"],
        "modules_imported": 2,
    }

    tool.assert_module_import_probe(payload)


def test_assert_module_import_probe_rejects_missing_modules():
    tool = _load_script()
    payload = {
        "failures": {},
        "modules": [],
        "modules_imported": 0,
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_module_import_probe(payload)

    assert "did not report modules" in str(excinfo.value)


def test_assert_module_import_probe_rejects_missing_failure_details():
    tool = _load_script()
    payload = {
        "failures": None,
        "modules": ["cli"],
        "modules_imported": 1,
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_module_import_probe(payload)

    assert "failure details" in str(excinfo.value)


def test_assert_module_import_probe_rejects_import_failures():
    tool = _load_script()
    payload = {
        "failures": {"codna.cli": "ImportError: boom"},
        "modules": ["cli"],
        "modules_imported": 1,
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_module_import_probe(payload)

    assert "module imports failed" in str(excinfo.value)


def test_assert_module_import_probe_rejects_count_mismatch():
    tool = _load_script()
    payload = {
        "failures": {},
        "modules": ["cli", "memory"],
        "modules_imported": 1,
    }

    with pytest.raises(tool.WheelSmokeError) as excinfo:
        tool.assert_module_import_probe(payload)

    assert "count does not match" in str(excinfo.value)
