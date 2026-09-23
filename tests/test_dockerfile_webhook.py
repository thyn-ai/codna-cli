"""Regression guards for the webhook image build.

The codna wheel build (cli/setup.py) bundles the agent-core sidecar INTO the wheel and fails
closed ("agent-core runtime source is missing run-server.mjs") unless the sibling agent-core/ is
present at build time. So in Dockerfile.webhook the agent-core COPY must land BEFORE
`pip install /app/cli`. This ordering broke once and failed the Fly build; lock it in.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "infra" / "docker" / "Dockerfile.webhook"
DOCKERIGNORE = ROOT / ".dockerignore"


def test_dockerfile_webhook_exists():
    assert DOCKERFILE.is_file(), "infra/docker/Dockerfile.webhook is required for the App-channel image"


def test_agent_core_copied_before_pip_install():
    """agent-core must be COPYed in before `pip install /app/cli`, or the wheel build fails closed."""
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    copy_idx = next(
        (i for i, line in enumerate(lines) if "agent-core" in line and line.strip().startswith("COPY")),
        None,
    )
    pip_idx = next(
        (i for i, line in enumerate(lines) if "pip install" in line and "/app/cli" in line),
        None,
    )
    assert copy_idx is not None, "expected a `COPY ... agent-core ...` line"
    assert pip_idx is not None, "expected a `pip install ... /app/cli` line"
    assert copy_idx < pip_idx, (
        "agent-core must be copied BEFORE `pip install /app/cli` — the codna wheel build bundles it "
        "and fails closed without the sibling agent-core/ present (this ordering broke the Fly build once)"
    )


def test_webhook_image_builds_agent_core_bundle_before_pip_install():
    """The wheel build requires both run-server.mjs and run-server.bundle.mjs in agent-core/."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "bun build vendor/cline/algenta/run-server.ts" in text
    assert "--outfile run-server.bundle.mjs" in text
    assert "cline.invalid" in text and "phc_redacted" in text
    assert text.index("--outfile run-server.bundle.mjs") < text.index("pip install --no-cache-dir /app/cli")


def test_webhook_image_is_keyless():
    """The webhook image must not bake in an LLM provider key."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    for banned in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        assert f"ENV {banned}" not in text and f"{banned}=" not in text, (
            f"{banned} must not be set in the webhook image — it stays keyless"
        )


def test_webhook_image_installs_runtime_port_inspection_tool():
    """codna doctor/start-stack needs lsof to enforce fixed local runtime ownership in-container."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    install_lines = " ".join(line.strip() for line in text.splitlines() if "apt-get install" in line or "lsof" in line)
    assert "lsof" in install_lines, (
        "Dockerfile.webhook must install lsof; without it `codna doctor --start-stack` fails "
        "with port_inspection_unavailable inside the deployed container"
    )


def test_webhook_image_installs_node_for_sidecar_supervisor():
    """run-server.mjs is launched with node and supervises the Bun sidecar child."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    install_lines = " ".join(
        line.strip()
        for line in text.splitlines()
        if "apt-get install" in line or "nodejs" in line
    )
    assert "nodejs" in install_lines, (
        "Dockerfile.webhook must install nodejs; without node, `codna doctor --start-stack` "
        "fails with node_runtime_not_found before the bundled Bun sidecar can start"
    )


def test_webhook_image_strips_generated_cli_build_artifacts_before_install():
    """A dirty local cli/build must not be repackaged into the Linux webhook image."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "rm -rf /app/cli/build" in text
    assert "/app/cli/*.egg-info" in text
    assert "pip install --no-cache-dir /app/cli" in text


def test_root_dockerignore_excludes_local_deploy_cruft():
    """Fly deploys from the repo root; local-only stores must never enter that context."""
    patterns = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required = {
        ".codex-worktrees",
        ".pnpm-store",
        ".venv",
        "**/node_modules",
        ".env",
        "*.pem",
        "**/build",
        "**/*.egg-info",
    }
    missing = sorted(required.difference(patterns))
    assert not missing, f".dockerignore is missing root-context exclusions: {missing}"
