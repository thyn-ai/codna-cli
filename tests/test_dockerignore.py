from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERIGNORE = ROOT / ".dockerignore"


def test_dockerignore_exists():
    assert DOCKERIGNORE.is_file(), "repo-root .dockerignore is required so the sidecar build context is slim"


def test_dockerignore_excludes_secrets():
    """A secret must never enter a Docker build context (it can be baked into an image layer)."""
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    for pattern in ("*.pem", "*.key", ".env", ".env.*", "keys.txt", ".keys.txt"):
        assert pattern in text, f".dockerignore must exclude {pattern}"


def test_dockerignore_excludes_bloat():
    """Excluding build/ + node_modules keeps the context small (the sidecar reinstalls deps in-image)."""
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    for pattern in ("build/", "**/node_modules", ".git", "mojo_build/", "mojo_env/"):
        assert pattern in text, f".dockerignore must exclude {pattern}"
