"""Guard package and GitHub Action contracts for local-first Codna installs."""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest

PYPROJECT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")
ROOT = Path(__file__).resolve().parents[2]
RELEASE_BUILDER = ROOT / "scripts" / "build_cli_release_dist.py"


def _data():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def _load_release_builder():
    spec = importlib.util.spec_from_file_location("build_cli_release_dist", RELEASE_BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wheel(path: Path, *, root_is_purelib: bool, tag: str, names: tuple[str, ...] = ()) -> None:
    metadata = "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: codna-test",
            f"Root-Is-Purelib: {str(root_is_purelib).lower()}",
            f"Tag: {tag}",
            "",
        ]
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("codna-0.1.0.dist-info/WHEEL", metadata)
        for name in names:
            archive.writestr(name, b"test")


def test_memory_extras_declared():
    project = _data()["project"]
    deps = project["dependencies"]
    extras = project["optional-dependencies"]
    assert "memory" in extras
    assert "memory-algenta" not in extras
    mem = " ".join(extras["memory"])
    base = " ".join(deps)
    assert "telys" in base and "numpy" in base and "tree-sitter" in base
    assert "faiss-cpu" in mem and "wordllama" in mem
    assert "telys" not in mem
    assert "telys-runtime" not in mem
    assert "telys-algenta" not in mem
    assert "memengine @" not in mem
    for extra_name, dependencies in extras.items():
        assert not any(" @ " in dependency or "git+" in dependency for dependency in dependencies), extra_name


def test_core_install_includes_local_telys_support_without_heavy_rerankers():
    deps = _data()["project"]["dependencies"]
    joined = " ".join(deps)
    assert "telys>=" in joined
    assert "tree-sitter>=" in joined
    assert "faiss" not in joined
    assert "wordllama" not in joined


def test_cli_readme_documents_base_codna_telys_install_contract():
    readme = (ROOT / "cli" / "README.md").read_text(encoding="utf-8")

    assert 'pip install "codna[memory]"' not in readme
    assert "pip install codna" in readme
    assert "no login and no extra install" in readme
    assert "Code memory and recall run on your machine" in readme


def test_core_install_uses_public_dependencies_only():
    data = _data()["project"]
    deps = data["dependencies"]
    extras = data["optional-dependencies"]
    assert any(dep.startswith("httpx") for dep in deps)
    # `mcp` is an OPTIONAL extra (it pulls a starlette/uvicorn web stack) so a plain
    # `pip install codna` stays lean — installed via `pip install codna[mcp]`.
    assert not any(dep.startswith("mcp") for dep in deps)
    assert "mcp>=1.2.0,<2" in extras["mcp"]
    assert any(dep.startswith("pyarrow") for dep in deps)
    assert any(dep.startswith("telys") for dep in deps)
    assert not any("algenta-sdk" in dep or "decision-engine" in dep for dep in deps)


def test_package_declares_local_readme_metadata():
    data = _data()["project"]
    assert data["readme"] == "README.md"
    assert data["description"] == (
        "Understand. Fix. Evolve. Codna maps your repository before it spends a token, reviews pull "
        "requests, fixes bugs and proves which scanner findings are reachable. Runs on your machine."
    )
    assert data["license"] == {"text": "Apache-2.0"}
    assert data["authors"] == [{"name": "Thyn"}]
    assert data["maintainers"] == [{"name": "Thyn"}]
    assert "repository-intelligence" in data["keywords"]
    assert "telys" in data["keywords"]
    assert "License :: OSI Approved :: Apache Software License" not in data["classifiers"]
    assert "Programming Language :: Python :: 3.13" in data["classifiers"]
    assert data["urls"]["Homepage"] == "https://codna.ai"
    assert data["urls"]["Documentation"] == "https://docs.codna.ai"
    assert data["urls"]["Source"] == "https://github.com/thyn-ai/codna"
    assert data["urls"]["Issues"] == "https://github.com/thyn-ai/codna/issues"
    assert data["urls"]["Security"] == "https://codna.ai/security"
    readme = (ROOT / "cli" / "README.md").read_text(encoding="utf-8")
    assert "pip install codna" in readme
    assert "`doctor`" in readme
    assert "Nothing else to install: no Node, Bun, Docker or server" in readme
    assert "codna mcp" in readme
    assert "Source distributions exclude runtime binaries, keys" in readme


def test_package_version_matches_runtime_version():
    data = _data()["project"]
    init_text = (ROOT / "cli" / "codna" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', init_text, re.MULTILINE)
    assert match is not None
    assert data["version"] == match.group(1)


def test_setup_bundles_agent_core_runtime_for_wheels_and_sdists():
    setup_py = (ROOT / "cli" / "setup.py").read_text(encoding="utf-8")
    assert "class build_py" in setup_py
    assert "class sdist" in setup_py
    assert "_copy_agent_core_runtime" in setup_py
    assert "codna" in setup_py and "agent-core" in setup_py
    assert "node_modules" in setup_py
    assert "run-server.bundle.mjs" in setup_py
    assert "missing run-server.mjs or run-server.bundle.mjs" in setup_py
    builder = (ROOT / "scripts" / "build_cli_release_dist.py").read_text(encoding="utf-8")
    assert "prepare_agent_core_bundle" in builder
    assert "sanitize_agent_core_bundle" in builder
    assert "run-server.bundle.mjs" in builder
    assert "cline.invalid" in builder
    assert "BUN_INSTALL_CACHE_DIR" in builder
    wheel_smoke = (ROOT / "scripts" / "verify_cli_wheel_tags.py").read_text(encoding="utf-8")
    assert "prepare_agent_core_bundle" in wheel_smoke
    assert "sanitize_agent_core_bundle" in wheel_smoke
    assert "run-server.bundle.mjs" in wheel_smoke


def test_cline_postbuild_is_concurrency_safe():
    script = ROOT / "agent-core" / "vendor" / "cline" / "algenta" / "postbuild.sh"
    processes = [
        subprocess.Popen(
            ["bash", str(script)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]

    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr

    cline_scope = ROOT / "agent-core" / "vendor" / "cline" / "node_modules" / "@cline"
    for package in ("shared", "llms", "agents", "core", "sdk"):
        dest = cline_scope / package
        assert dest.is_symlink()
        assert os.readlink(dest) == f"../../sdk/packages/{package}"


def test_setuptools_includes_runtime_package():
    packages = _data()["tool"]["setuptools"]["packages"]
    assert "codna" in packages
    assert "codna.runtime" in packages


def test_telys_package_data_is_manifest_scoped():
    package_data = _data()["tool"]["setuptools"]["package-data"]["codna"]
    assert package_data == [
        "_telys_runtime/libame_kernel.*",
        "_telys_runtime/lib*.dylib",
        "_telys_runtime/lib*.so",
        "_telys_runtime/*.dll",
        "_telys_runtime/manifest.json",
        # Codna's baked-in OEM entitlement (never committed; staged into the wheel at release time).
        "_telys_runtime/oem-license.jwt",
        # The self-contained agent-core sidecar (bun build --compile), staged per-platform at release.
        "_agent_core_runtime/codna-sidecar",
    ]
    assert "_telys_runtime/*" not in package_data
    assert "_agent_core_runtime/*" not in package_data


def test_sdist_manifest_excludes_local_runtime_and_secret_artifacts():
    manifest = (ROOT / "cli" / "MANIFEST.in").read_text(encoding="utf-8")
    assert (
        "recursive-exclude codna/_telys_runtime "
        "libame_kernel.* lib*.dylib lib*.so *.dll manifest.json oem-license.jwt"
    ) in manifest
    # The baked OEM license is a credential-bearing wheel artifact; it must never enter the source sdist.
    assert "exclude codna/_telys_runtime/oem-license.jwt" in manifest
    for pattern in ("keys.txt", ".keys.txt", ".env", ".env.*", "*.log", "*.pem", "*.key", "*.jwt"):
        assert pattern in manifest
    assert "local-stack.json" in manifest
    assert "launcher.state" in manifest


def test_native_runtime_build_has_wheel_tag_guard():
    data = _data()
    assert "wheel>=0.43" in data["build-system"]["requires"]
    assert (ROOT / "cli" / "setup.py").is_file()


def test_github_action_preserves_codna_failure_status_without_forcing_remote_engine():
    action = (ROOT / "action.yml").read_text(encoding="utf-8")
    assert 'default: ""' in action
    assert 'package_spec="${CODNA_ACTION_PACKAGE_SPEC:-codna}"' in action
    assert 'package_spec="codna==$version"' in action
    assert 'pipx install --pip-args=--only-binary=codna --force "$package_spec"' in action
    assert 'python3 -m pip install --user --upgrade --only-binary=codna "$package_spec"' in action
    assert 'echo "$HOME/.local/bin" >> "$GITHUB_PATH"' in action
    assert 'package_path="$GITHUB_ACTION_PATH/cli"' not in action
    assert "bun run build:sdk" not in action
    assert "bash algenta/postbuild.sh" not in action
    assert "run-server.bundle.mjs" not in action
    assert 'CODNA_SIDECAR_DIR=$GITHUB_ACTION_PATH/agent-core' not in action
    assert "engine-url:" not in action
    assert "CODNA_ACTION_ENGINE_URL" not in action
    assert "export CODNA_ENGINE_URL" not in action
    assert 'repo_url="https://github.com/${{ github.repository }}.git"' not in action
    assert 'out="$(codna "${args[@]}" 2>&1)" || true' not in action
    assert "status=$?" in action
    assert 'exit "$status"' in action
    # fix mode parses the PR URL from the --json result's top-level field (with a regex fallback),
    # not by scraping human output; runtime logs are kept on stderr so stdout is clean JSON.
    # ...and the output is written as exactly one line: an embedded newline in the value can never
    # become a second GITHUB_OUTPUT key.
    assert 'url="$(printf \'%s\' "$url" | tr -d \'\\r\\n\')"' in action
    assert 'printf \'pull_request_url=%s\\n\' "$url" >> "$GITHUB_OUTPUT"' in action
    assert 'echo "pull_request_url=$url"' not in action
    assert '2>/tmp/codna.err' in action


def test_github_action_exposes_security_autofix_mode():
    """The security-autofix loop must be reachable via the Action channel, not just the CLI."""
    action = (ROOT / "action.yml").read_text(encoding="utf-8")
    # secure-mode inputs are declared
    for token in ("mode:", "from-sarif:", "reachability-engine:", "verification:"):
        assert token in action, f"action.yml is missing the security-autofix input {token!r}"
    # secure mode runs `codna secure --from-sarif ...`
    assert 'CODNA_ACTION_MODE: ${{ inputs.mode }}' in action
    assert 'if [ "$CODNA_ACTION_MODE" = "secure" ]; then' in action
    assert 'args=(secure "$repo_url" --ref "$CODNA_ACTION_SHA" --from-sarif "$CODNA_ACTION_FROM_SARIF"' in action
    # secure mode is READ-ONLY: it must never open PRs from this single (token-holding) step —
    # the security spec requires the analysis worker to hold no write token. PRs use the
    # privilege-separated two-job examples/codna-secure.yml instead.
    assert "--open-pr" not in action.split('if [ "$CODNA_ACTION_MODE" = "secure" ]; then', 1)[1].split("else", 1)[0]
    assert 'args+=(--verification "$CODNA_ACTION_VERIFICATION")' in action
    # fix mode is preserved as the default and still opens a PR (now with --json for a stable,
    # machine-parseable result — the action reads pull_request_url from it, not a regex).
    assert 'default: "fix"' in action
    assert 'args=(fix "$repo_url" --ref "$CODNA_ACTION_SHA" --open-pr --json --model "$CODNA_ACTION_MODEL")' in action


def test_ci_workflow_uses_node24_compatible_pinned_actions():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # Each required action must be PRESENT and pinned to a full 40-hex commit SHA — assert by action
    # NAME, not an exact SHA, so a Dependabot version bump (which changes the pinned SHA) keeps the
    # supply-chain guarantee without breaking this test on every bump. The blanket "every uses: is
    # SHA-pinned" check below is the hard guarantee; the negative asserts guard known node<24 refs.
    # No actions/setup-python: the fleet is self-hosted macOS, where that action hardcodes the
    # GitHub-hosted /Users/runner/hostedtoolcache path and can never find a local tool cache --
    # Python comes from the pre-provisioned runner tool cache via PATH instead. The blanket
    # "every uses: is SHA-pinned" check below still enforces the supply-chain guarantee.
    required_actions = {
        "actions/checkout",
        "docker/setup-buildx-action",
        "docker/login-action",
        "docker/metadata-action",
        "docker/build-push-action",
    }
    for action in required_actions:
        assert re.search(rf"{re.escape(action)}@[0-9a-f]{{40}}(?:\s|$)", workflow), (
            f"{action} must be present in ci.yml and pinned to a full 40-hex commit SHA"
        )

    uses_lines = [line.strip() for line in workflow.splitlines() if line.strip().startswith("uses: ")]
    unpinned_lines = [line for line in uses_lines if not re.search(r"@[0-9a-f]{40}(?:\s|$)", line)]
    assert unpinned_lines == []
    # Regression guards against specific known node<24 (or otherwise-superseded) refs — a downgrade to
    # any of these must fail even though they are SHA-pinned.
    assert "actions/checkout@v4" not in workflow
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" not in workflow
    assert "docker/build-push-action@ca052bb54ab0790a636c9b5f226502c73d547a25" not in workflow


def test_sidecar_image_publishes_under_current_github_owner():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ghcr.io/thyn-ai/codna/sidecar" in workflow
    assert "ghcr.io/escrypto-labs/codna/sidecar" not in workflow


def test_public_action_examples_use_current_github_owner():
    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "configuration.md",
        ROOT / "docs" / "github-action.md",
        ROOT / "examples" / "codna-autofix.yml",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "thyn-ai/codna-action@v1" in combined
    assert "uses: thyn-ai/codna@v1" not in combined
    assert "escrypto-labs/codna@v1" not in combined
    assert "codna-ai/fix-action" not in combined


def test_release_runbook_documents_exact_pypi_trusted_publisher_claims():
    runbook = (ROOT / "docs-internal" / "release-runbook.md").read_text(encoding="utf-8")

    required = [
        "Project name: codna",
        "Owner: thyn-ai",
        "Repository name: codna",
        "Workflow filename: publish-cli.yml",
        "Environment name: pypi",
        "aud: pypi",
        "repository: thyn-ai/codna",
        "repository_owner: thyn-ai",
        "ref: refs/heads/main",
        "workflow_ref: thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
        "job_workflow_ref: thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
        "environment: pypi",
        "sub: repo:thyn-ai/codna:environment:pypi",
    ]
    for value in required:
        assert value in runbook
    assert "Workflow name: publish-cli.yml" not in runbook


def test_cli_publish_workflow_uses_trusted_publishing_and_contract_verification():
    workflow = (ROOT / ".github" / "workflows" / "publish-cli.yml").read_text(encoding="utf-8")
    assert "build-verify:" in workflow
    assert "type: boolean" in workflow
    assert "default: false" in workflow
    assert "default: true" in workflow
    assert "verify_pypi_oidc:" in workflow
    assert "verify-pypi-oidc:" in workflow
    assert "verify PyPI publish auth" in workflow
    assert "inputs.verify_pypi_oidc == true && inputs.publish != true" in workflow
    assert "needs: build-verify" in workflow
    # workflow_dispatch-only trigger: release-please dispatches this itself with publish=true
    # right after creating the tag + Release, so there's exactly one publish attempt per release,
    # never two racing (an `on: release: types: [published]` trigger here used to fire alongside
    # it, and whichever lost the race always failed with a duplicate-upload 400 from PyPI).
    assert "on:\n  workflow_dispatch:" in workflow
    assert "on:\n  release:" not in workflow
    assert "if: ${{ inputs.publish == true }}" in workflow
    assert "environment: pypi" in workflow
    assert "id-token: write" in workflow
    assert "uses: oven-sh/setup-bun@0c5077e51419868618aeaa5fe8019c62421857d6" in workflow
    assert "bun-version: 1.3.13" in workflow
    assert "python -m pip install --upgrade build twine" in workflow
    assert "python -m pip install --upgrade build cryptography setuptools twine wheel" in workflow
    assert "python scripts/build_cli_release_dist.py --cli-dir cli" in workflow
    assert "python scripts/download_telys_runtime_bundle.py" in workflow
    assert "python scripts/stage_codna_telys_runtime.py" in workflow
    assert "python scripts/verify_pypi_trusted_publisher.py" in workflow
    assert "Select dry-run publish auth" in workflow
    assert "PYPI_DRY_RUN_AUTH=token" in workflow
    assert "PYPI_DRY_RUN_AUTH=oidc" in workflow
    assert "Verify PyPI API token fallback" in workflow
    assert "Verify trusted publisher before staging runtimes" in workflow
    assert workflow.index("Verify trusted publisher before staging runtimes") < workflow.index("Stage Linux Telys runtime")
    assert "CODNA_TELYS_RUNTIME_LINUX_X86_64_BUNDLE_URL" in workflow
    assert "CODNA_TELYS_RUNTIME_MACOS_ARM64_BUNDLE_URL" in workflow
    # Codna publishes one user-facing package. Platform wheels fetch signed Telys runtime bundles
    # with a release-only OEM download token, then stage Codna's offline OEM license into the wheel.
    assert "CODNA_TELYS_API_KEY" not in workflow
    assert "secrets.CODNA_TELYS_OEM_DOWNLOAD_TOKEN" in workflow
    assert "secrets.CODNA_TELYS_LICENSE_JWT" in workflow
    assert "CODNA_TELYS_RUNTIME_DOWNLOAD_TOKEN" not in workflow
    assert "https://packages.telys.ai/runtime/latest/linux-x86_64.bundle" in workflow
    assert "https://packages.telys.ai/runtime/latest/macos-arm64.bundle" in workflow
    assert "--target-platform linux-x86_64" in workflow
    assert "--target-platform macos-arm64" in workflow
    assert "--platform-tag manylinux_2_28_x86_64" in workflow
    assert "--platform-tag linux_x86_64" not in workflow
    assert "--platform-tag macosx_14_0_arm64" in workflow
    assert "--skip-sdist" in workflow
    assert "--no-clean" in workflow
    assert "python scripts/verify_cli_wheel_tags.py --cli-dir cli" in workflow
    assert "python -m twine check cli/dist/*" in workflow
    assert "actions/upload-artifact" not in workflow
    assert "actions/download-artifact" not in workflow
    assert "python -m build cli" not in workflow
    # 3 = build-verify base metadata build + Linux platform wheel + macOS platform wheel.
    assert workflow.count("python scripts/build_cli_release_dist.py") == 3
    assert workflow.count("uses: oven-sh/setup-bun@0c5077e51419868618aeaa5fe8019c62421857d6") == 2
    assert workflow.count("python scripts/verify_pypi_trusted_publisher.py") == 2
    assert workflow.count("python scripts/download_telys_runtime_bundle.py") == 2
    assert workflow.count("--token-env CODNA_TELYS_OEM_DOWNLOAD_TOKEN") == 2
    assert workflow.count("python scripts/stage_codna_telys_oem_license.py") == 2
    assert workflow.count("--require-staged-runtime") == 2
    # The wheel's memengine must come from the staged SIGNED bundle (option-B), never the repo's
    # vendored copy — and each sync must run AFTER its staging step but BEFORE the dist build.
    assert workflow.count("python scripts/sync_wheel_memengine_from_staged_runtime.py --cli-dir cli") == 2
    assert "Sync wheel memengine from the staged signed bundle (linux)" in workflow
    assert "Sync wheel memengine from the staged signed bundle (macos)" in workflow
    assert workflow.index("Stage Linux Telys runtime") < workflow.index(
        "Sync wheel memengine from the staged signed bundle (linux)") < workflow.index("Build Linux distribution")
    assert workflow.index("Stage macOS Telys runtime") < workflow.index(
        "Sync wheel memengine from the staged signed bundle (macos)") < workflow.index("Build macOS wheel")
    assert "base_wheel_only" not in workflow
    assert "telys runtime install" not in workflow
    assert workflow.count("python scripts/verify_cli_wheel_tags.py --cli-dir cli") == 2
    assert workflow.count("python -m twine check cli/dist/*") == 2
    # Pinned to a full 40-hex commit SHA, never a floating tag: a mutable ref on the action that
    # uploads releases is a supply-chain hole. Asserted as a PROPERTY, not as one literal SHA --
    # a hardcoded SHA fails on every LEGITIMATE re-pin (this one broke when the org aligned action
    # refs org-wide to stop self-hosted runner cache thrash), and the only available fix is "bump
    # the constant", which is not review and provides no real protection.
    pypi_pins = re.findall(r"pypa/gh-action-pypi-publish@(\S+)", workflow)
    assert pypi_pins, "the OIDC publish path must use pypa/gh-action-pypi-publish"
    for _ref in pypi_pins:
        assert re.fullmatch(r"[0-9a-f]{40}", _ref), (
            f"pypa/gh-action-pypi-publish must be pinned to a full commit SHA, got {_ref!r}"
        )
    # Publish auth policy: trusted-publisher OIDC is PREFERRED; a GATED PyPI API-token fallback is
    # allowed so a release does not hard-block on trusted-publisher registration (which is a one-time
    # PyPI UI step). The token/password credential is confined to the token-gated publish step
    # (PUBLISH_AUTH == 'token') and the OIDC path carries no password. See publish-cli.yml
    # "Select publish auth (token fallback vs trusted-publisher OIDC)". The token path uploads with
    # NATIVE twine: pypa/gh-action-pypi-publish is a Linux-only container action and hard-refuses
    # the macOS fleet (run 31986507815), so only the OIDC path may use it.
    assert "PUBLISH_AUTH=token" in workflow and "PUBLISH_AUTH=oidc" in workflow
    assert "env.PUBLISH_AUTH == 'oidc'" in workflow
    assert "env.PUBLISH_AUTH == 'token'" in workflow
    assert "TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}" in workflow
    # IDEMPOTENT upload on BOTH auth paths. PyPI versions are immutable, so re-uploading one it
    # already has can never overwrite anything -- "already there" IS the desired end state. Without
    # these flags a duplicate publish fails on "File already exists" and turns a correct, published
    # release into a red deployment: it happened with an `on: release` trigger racing
    # workflow_dispatch, and again on 0.2.7 when a CDN-stale PyPI read made a release run dispatch a
    # second publish (deployment 5974216490 succeeded, duplicate 5974321714 failed, 0.2.7 live
    # throughout). Asserted so the idempotency cannot be dropped again.
    assert "python -m twine upload --non-interactive --skip-existing cli/dist/*" in workflow
    # The OIDC path is asserted STRUCTURALLY, on the parsed step input -- a substring check for
    # "skip-existing: true" is satisfied by the prose comment above those steps, so it could not
    # fail (verified: deleting the real input left a text-only assert green).
    import yaml as _yaml

    _doc = _yaml.safe_load(workflow)
    _oidc = [
        st
        for job in _doc["jobs"].values()
        for st in job.get("steps", [])
        if "gh-action-pypi-publish" in str(st.get("uses", ""))
    ]
    assert _oidc, "the OIDC publish step vanished"
    for st in _oidc:
        assert (st.get("with") or {}).get("skip-existing") is True, (
            f"{st.get('name')!r} must pass skip-existing: true so a duplicate upload is a no-op"
        )
    assert workflow.count("TWINE_PASSWORD:") == 1  # ONLY the gated token-fallback step, nowhere else
    assert "password:" not in workflow  # the token path must not use the pypa action's password input
    assert "api-token:" not in workflow


def test_docs_audit_supports_manual_release_dispatch():
    workflow = (ROOT / ".github" / "workflows" / "docs-audit.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert 'paths: ["docs/**", "gitbook/**", ".gitbook.yaml"]' not in workflow


def test_public_installer_uses_packaged_cli_runtime_only():
    installer = (ROOT / "install.sh").read_text(encoding="utf-8")

    assert "need docker" not in installer
    assert "docker compose" not in installer
    assert "CODNA_ENGINE_URL=http://127.0.0.1" not in installer
    assert "ALGENTA_AGENT_CORE_URL=http://127.0.0.1" not in installer
    assert "pipx install --force \"$CODNA_PACKAGE_SPEC\"" in installer
    assert "python3 -m pip install --user --upgrade \"$CODNA_PACKAGE_SPEC\"" in installer
    assert "autostarts on first use" in installer.lower()


def test_release_dist_builder_fails_closed_without_staged_runtime(tmp_path: Path):
    tool = _load_release_builder()

    with pytest.raises(tool.ReleaseDistError) as excinfo:
        tool.verify_staged_runtime(tmp_path / "missing-runtime", target_platform="linux-x86_64")

    assert "staged Telys runtime is required for publish" in str(excinfo.value)


def test_release_dist_builder_fails_closed_without_staged_oem_license(tmp_path: Path):
    tool = _load_release_builder()
    runtime_dir = tmp_path / "runtime"
    kernel = runtime_dir / tool.expected_artifact_name("macos-arm64")
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"kernel")
    (runtime_dir / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "artifact": kernel.name,
            "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
            "size_bytes": kernel.stat().st_size,
            "platform": "macos-arm64",
            "source_basename": kernel.name,
        }),
        encoding="utf-8",
    )

    with pytest.raises(tool.ReleaseDistError) as excinfo:
        tool.verify_staged_runtime(runtime_dir, target_platform="macos-arm64")

    assert "OEM Telys license is required" in str(excinfo.value)


def test_release_dist_builder_rejects_pure_publish_wheel(tmp_path: Path):
    tool = _load_release_builder()
    wheel = tmp_path / "codna-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, root_is_purelib=True, tag="py3-none-any")

    metadata = tool.inspect_wheel(wheel)

    with pytest.raises(tool.ReleaseDistError) as excinfo:
        tool.assert_native_runtime_wheel(metadata, target_platform="macos-arm64")

    assert "must not be py3-none-any" in str(excinfo.value)


def test_release_dist_builder_accepts_native_runtime_wheel(tmp_path: Path):
    tool = _load_release_builder()
    wheel = tmp_path / "codna-0.1.0-py3-none-macosx_15_0_arm64.whl"
    _write_wheel(
        wheel,
        root_is_purelib=False,
        tag="py3-none-macosx_15_0_arm64",
        names=(
            f"codna/_telys_runtime/{tool.expected_artifact_name('macos-arm64')}",
            "codna/_telys_runtime/manifest.json",
            "codna/_telys_runtime/oem-license.jwt",
            "codna/_agent_core_runtime/codna-sidecar",
        ),
    )

    tool.assert_native_runtime_wheel(tool.inspect_wheel(wheel), target_platform="macos-arm64")


@pytest.mark.parametrize(
    "artifact_name",
    [
        "libAsyncRTMojoBindings.dylib",
        "libty_runtime.so",
        "ame_kernel.dll",
        "oem-license.jwt",
    ],
)
def test_release_dist_builder_rejects_runtime_artifacts_in_sdist(tmp_path: Path, artifact_name: str):
    tool = _load_release_builder()
    sdist = tmp_path / "codna-0.1.0.tar.gz"
    payload = b"native"
    member_name = f"codna-0.1.0/codna/_telys_runtime/{artifact_name}"

    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(tool.ReleaseDistError) as excinfo:
        tool.assert_sdist_excludes_runtime(sdist)

    assert member_name in str(excinfo.value)
