"""Local and HTTP repository-path validation for durable runs."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .errors import DurableCliError

__all__ = ["existing_git_repo", "http_repo_dir"]


def existing_git_repo(resolved_text: str) -> Path:
    resolved = Path(resolved_text)
    if not resolved.is_dir():
        raise DurableCliError(
            f"repo directory does not exist: {resolved}",
            code="durable_repo_missing",
            details={"repo_dir": str(resolved)},
        )
    return _git_work_tree(resolved)


def http_repo_dir(raw: str | None, *, allowed_roots: tuple[Path, ...]) -> Path:
    """Resolve an HTTP-supplied path only inside an operator-approved root."""
    if raw and "\0" in raw:
        raise DurableCliError(
            "repo directory contains a NUL byte.",
            code="durable_repo_missing",
            details={"repo_dir": "<NUL>"},
        )
    resolved_text = os.path.realpath(os.path.expanduser(raw)) if raw else str(Path.cwd())
    for root in allowed_roots:
        root_text = os.path.realpath(str(root))
        if resolved_text == root_text:
            return existing_git_repo(root_text)
        root_prefix = root_text if root_text.endswith(os.sep) else f"{root_text}{os.sep}"
        if resolved_text.startswith(root_prefix):
            resolved = Path(resolved_text)
            if not resolved.is_dir():
                raise DurableCliError(
                    f"repo directory does not exist: {resolved}",
                    code="durable_repo_missing",
                    details={"repo_dir": str(resolved)},
                )
            return _git_work_tree(resolved)
    raise DurableCliError(
        f"repo directory is outside the operator-approved roots: {resolved_text}",
        code="durable_repo_outside_allowed_roots",
        details={"repo_dir": resolved_text, "allowed_roots": [str(root) for root in allowed_roots]},
        hint="set CODNA_DURABLE_REPO_ROOTS to the workspace roots this server may edit",
    )


def _git_work_tree(resolved: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != "true":
        raise DurableCliError(
            f"not a git work tree: {resolved}",
            code="durable_repo_not_git",
            details={"repo_dir": str(resolved), "stderr": (completed.stderr or "").strip()[:400]},
            hint="a durable run edits a checkout and commits a ref per turn, so it needs git",
        )
    return resolved
