"""Bridge: hand the IDE the SKILL catalogue and one skill's body as JSON.

The sibling of :mod:`codna.rules_bridge`, and deliberately the same shape: one JSON object in, one JSON
object out, the SAME ``codna.rules`` engine answering. It exists because the renderer needs the
catalogue as STRUCTURE (to offer ``/skill-name`` completion) rather than as prompt text, and
re-implementing skill discovery in TypeScript would be the split-reader defect this package removes.

**Protocol.**

Request on stdin, one of::

    {"repo": "/abs/path", "files_in_context": ["src/a.ts"] | null}      → the catalogue
    {"repo": "/abs/path", "skill": "pdf"}                              → one skill's body

Response on stdout::

    {"ok": true, "skills": [...], "skills_dropped": [...], "warnings": [...],
     "used_bytes": 123, "budget_bytes": 4096}                          catalogue
    {"ok": true, "name": "pdf", "body": "...", "bundled": [...], ...}  one skill

On a refusal the response is ``{"ok": false, "error": {"code": …, "message": …, "details": {}}}`` and
the exit code is 1. The process NEVER guesses: an absent ``repo`` is a refusal, not "use the cwd".

**Why a second bridge and not a flag on the rules one.** ``rules_bridge`` answers with a PROMPT STRING
plus the account of what went into it, and its whole contract is about bytes fitting a budget.
This one answers with the catalogue as DATA — the composer does not parse prompt text, and a consumer
that had to would be reading a format that is not a protocol. Both call the same Python functions.
"""
from __future__ import annotations

import json
import sys

__all__ = ["handle", "main"]

# The composer's catalogue ceiling. Smaller than the prompt's 8 KB on purpose: this answer feeds a `/`
# completion list, and a menu longer than a few dozen entries is not a menu a user reads.
DEFAULT_MAX_BYTES = 4_096


def _error(code: str, message: str, **details) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "details": details}}


def handle(request: object) -> dict:
    """Answer one bridge request. Pure apart from reading the repository."""
    if not isinstance(request, dict):
        return _error("bridge_bad_request", "the request must be a JSON object")
    repo = request.get("repo")
    if not isinstance(repo, str) or not repo.strip():
        return _error("bridge_bad_request", "`repo` is required and must be a non-empty string")

    from .rules import discover_skills, load_skill, skill_catalogue

    # The PULL: one skill's body. Requested by name, which is what `/skill-name` resolves.
    raw_name = request.get("skill")
    if raw_name is not None:
        if not isinstance(raw_name, str) or not raw_name.strip():
            return _error("bridge_bad_request", "`skill` must be a non-empty string")
        home = request.get("home")
        if home is not None and (not isinstance(home, str) or not home.strip()):
            return _error("bridge_bad_request", "`home` must be a non-empty string or null")
        skill = load_skill(repo, raw_name, home=home)
        if skill is None:
            return _error(
                "skill_not_found",
                f"no skill named {raw_name!r}",
                requested=raw_name,
                repo=repo,
            )
        return {"ok": True, **_skill(skill), "body": skill.body}

    raw_files = request.get("files_in_context")
    if raw_files is not None and not isinstance(raw_files, list):
        return _error("bridge_bad_request", "`files_in_context` must be a list of strings or null")
    files = None if raw_files is None else [str(item) for item in raw_files]

    # The HOME directory the user layer is read from. Absent means the real one — a product host wants
    # `~/.cursor/skills` exactly as Cursor discovers it — and an explicit value lets a test (or a host
    # scoping the machine layer deliberately) choose.
    home = request.get("home")
    if home is not None and (not isinstance(home, str) or not home.strip()):
        return _error("bridge_bad_request", "`home` must be a non-empty string or null")

    max_bytes = request.get("max_bytes", DEFAULT_MAX_BYTES)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        return _error("bridge_bad_request", "`max_bytes` must be an integer >= 0")

    discovery = discover_skills(repo, home=home)
    result = skill_catalogue(discovery, files_in_context=files, max_bytes=max_bytes)
    return {
        "ok": True,
        "skills": [_skill(skill) for skill in result.catalogued],
        "skills_dropped": [
            {
                "name": entry.name,
                "source_path": entry.source_path,
                "reason": "budget",
                "bytes_needed": entry.bytes,
            }
            for entry in result.dropped
        ],
        "warnings": [
            {"code": warning.code, "source_path": warning.source_path, "message": warning.message}
            for warning in discovery.warnings
        ],
        "used_bytes": result.used_bytes,
        "budget_bytes": result.budget_bytes,
        "files_in_context": list(files or ()),
        "files_known": files is not None,
    }


def _skill(skill) -> dict:
    """One skill WITHOUT its body, unless the caller asked for the pull.

    ``bundled`` is relative to the skill folder, so a caller can open `scripts/x.py` without knowing
    where the repository is checked out.
    """
    return {
        "name": skill.name,
        "description": skill.description,
        "scope": skill.scope.value,
        "source_path": skill.source_path,
        "paths": list(skill.paths),
        "mode": skill.mode.value,
        "bundled": list(skill.bundled),
    }


def main(argv: list[str] | None = None) -> int:
    """Read one request from stdin, write one response to stdout. Returns the exit code."""
    del argv  # the protocol is stdin/stdout only; flags would be a second, drifting interface
    try:
        raw = sys.stdin.read()
    except OSError as exc:  # pragma: no cover - stdin closed by the parent
        print(json.dumps(_error("bridge_stdin_failed", str(exc))), file=sys.stdout)
        return 1
    try:
        request = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps(_error("bridge_bad_json", f"cannot parse the request: {exc}")), file=sys.stdout)
        return 1

    try:
        response = handle(request)
    except Exception as exc:  # noqa: BLE001 — a bridge must answer, never traceback into the host
        response = _error("bridge_failed", f"{type(exc).__name__}: {exc}")
    print(json.dumps(response))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    sys.exit(main())
