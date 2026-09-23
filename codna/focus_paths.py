"""Focus-path extraction — pull explicit file paths out of an issue string to steer the engine
toward the right files. Pure helpers split out of cli.py to keep it under the modularity ceiling."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re


_FOCUS_PATH_PATTERNS = (
    re.compile(r'File\s+"([^"]+)"(?:,\s*line\s*\d+)?', re.I),
    re.compile(r"\b(?:Target file|Path):\s*`?([^`\s]+)`?", re.I),
)


def _normalize_focus_path(local_path: str, raw_path: str) -> str | None:
    repo_root = Path(local_path).expanduser().resolve()
    candidate = raw_path.strip().strip("`'\"")
    candidate = candidate.lstrip("<([{")
    candidate = candidate.rstrip(">)]},:;")
    candidate = candidate.replace("\\", "/")
    if not candidate:
        return None
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        try:
            relative = candidate_path.resolve().relative_to(repo_root)
        except ValueError:
            return None
    else:
        relative = Path(PurePosixPath(candidate))
        if ".." in relative.parts or relative.is_absolute():
            return None
    normalized = PurePosixPath(relative.as_posix())
    if normalized.as_posix() in {"", "."}:
        return None
    if not (repo_root / normalized.as_posix()).is_file():
        return None
    return normalized.as_posix()


def _issue_focus_paths(local_path: str, issue: str | None) -> list[str]:
    if not issue:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for pattern in _FOCUS_PATH_PATTERNS:
        for match in pattern.finditer(issue):
            normalized = _normalize_focus_path(local_path, match.group(1))
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
            if len(output) >= 8:
                return output
    return output


