"""Bridge: hand another process (the IDE backend) the rule engine's answer as JSON.

The IDE does not link Python, and re-implementing the rule engine in TypeScript is exactly the
"two readers that disagree" defect this work removes — a second implementation would drift from this
one the first time either changed. So the IDE asks this bridge, which is a thin stdin/stdout wrapper
over the SAME ``codna.rules`` call ``codna fix`` and durable runs use.

**Protocol (one JSON object in, one JSON object out).**

Request on stdin::

    {"repo": "/abs/path", "files_in_context": ["src/a.tsx"] | null,
     "explicit_mentions": ["@python"] | null, "max_bytes": 8000}

Response on stdout::

    {"ok": true, "text": "...", "kept": [...], "catalogued": [...],
     "dropped": [...], "truncated": [...], "warnings": [...]}

On a refusal (bad request, unreadable repo) the response is
``{"ok": false, "error": {"code": ..., "message": ..., "details": {...}}}`` and the exit code is 1.
The process NEVER guesses: an absent ``repo`` is a refusal, not "use the cwd", because the cwd of a
spawned bridge is not the user's repository and silently reading the wrong tree's rules is worse than
an error the host can log.

**Why a subprocess is acceptable here.** It spawns once per session create (not per turn), costs a
Python start (~0.2 s measured) and produces text the session then carries. The alternative — a
long-lived RPC — would add a lifecycle, a port and a failure mode for a call made once per chat.
It is also the pattern the IDE already uses for its Python neighbours (`telysHealth.ts` spawns
``python -m codna_parity.telys_worker``).
"""
from __future__ import annotations

import json
import sys

__all__ = ["handle", "main"]

# The IDE's prompt ceiling, matching `durable.rules.MAX_RULES_CHARS`. Kept as a literal here because
# this module is spawned with no package context on `sys.path` guarantees; the caller may override it.
DEFAULT_MAX_BYTES = 8_000


def _error(code: str, message: str, **details) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "details": details}}


def handle(request: object) -> dict:
    """Answer one bridge request. Pure apart from reading the repository."""
    if not isinstance(request, dict):
        return _error("bridge_bad_request", "the request must be a JSON object")
    repo = request.get("repo")
    if not isinstance(repo, str) or not repo.strip():
        return _error("bridge_bad_request", "`repo` is required and must be a non-empty string")

    raw_files = request.get("files_in_context")
    if raw_files is not None and not isinstance(raw_files, list):
        return _error("bridge_bad_request", "`files_in_context` must be a list of strings or null")
    files = None if raw_files is None else [str(item) for item in raw_files]

    raw_mentions = request.get("explicit_mentions")
    if raw_mentions is not None and not isinstance(raw_mentions, list):
        return _error("bridge_bad_request", "`explicit_mentions` must be a list of strings or null")
    mentions = None if raw_mentions is None else [str(item) for item in raw_mentions]

    raw_skills = request.get("include_skills", True)
    if not isinstance(raw_skills, bool):
        return _error("bridge_bad_request", "`include_skills` must be true or false")

    # The HOME directory the user layer is read from. Absent means the real one — which is what a
    # product host wants, because `~/.cursor/skills` is a directory Cursor documents as auto-discovered
    # — and an explicit value is for a host that scopes the machine layer deliberately, and for tests
    # that must not depend on what happens to be in the running machine's home.
    home = request.get("home")
    if home is not None and (not isinstance(home, str) or not home.strip()):
        return _error("bridge_bad_request", "`home` must be a non-empty string or null")

    max_bytes = request.get("max_bytes", DEFAULT_MAX_BYTES)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        return _error("bridge_bad_request", "`max_bytes` must be an integer >= 0")

    # ONE reader for both halves. `load_context` renders the rules and the skills catalogue together
    # under ONE budget, which is why this bridge did not grow a second call: a second call would be a
    # second ceiling, and two ceilings can each be satisfied while the pair overflows the prompt.
    from .rules import load_context

    loaded = load_context(
        repo,
        files_in_context=files,
        explicit_mentions=mentions,
        max_bytes=max_bytes,
        home=home,
        include_skills=raw_skills,
    )
    result = loaded.rules.result
    selection = loaded.rules.selection
    return {
        "ok": True,
        "text": loaded.text,
        "used_bytes": loaded.used_bytes,
        "budget_bytes": loaded.budget_bytes,
        "files_in_context": list(selection.files),
        "files_known": selection.files_known,
        "kept": [_rule(rule) for rule in result.kept],
        "catalogued": [_rule(rule) for rule in result.catalogued],
        "skills": [_skill(skill) for skill in loaded.skill_catalogue.catalogued],
        "skills_dropped": [_catalogue_entry(entry) for entry in loaded.skill_catalogue.dropped],
        "skills_used_bytes": loaded.skill_catalogue.used_bytes,
        "skills_budget_bytes": loaded.skill_catalogue.budget_bytes,
        "dropped": [
            {
                "name": dropped.name,
                "source_path": dropped.source_path,
                "reason": dropped.reason,
                "bytes_needed": dropped.bytes_needed,
            }
            for dropped in result.dropped
        ],
        "truncated": [
            {
                "name": record.name,
                "source_path": record.source_path,
                "bytes_kept": record.bytes_kept,
                "bytes_total": record.bytes_total,
            }
            for record in result.truncated
        ],
        "warnings": [
            {"code": warning.code, "source_path": warning.source_path, "message": warning.message}
            for warning in loaded.warnings
        ],
    }


def _rule(rule) -> dict:
    return {
        "name": rule.name,
        "mode": rule.mode.value,
        "scope": rule.scope.value,
        "source_path": rule.source_path,
        "base_dir": rule.base_dir,
        "description": rule.description,
        "globs": list(rule.globs),
    }


def _skill(skill) -> dict:
    return {
        "name": skill.name,
        "description": skill.description,
        "scope": skill.scope.value,
        "source_path": skill.source_path,
        "paths": list(skill.paths),
        "mode": skill.mode.value,
        # WHICH FILES, not their contents: the catalogue's job is to say the skill exists and where its
        # material lives, so a caller can decide to pull it. Bodies never travel here.
        "bundled": list(skill.bundled),
    }


def _catalogue_entry(entry) -> dict:
    return {
        "name": entry.name,
        "source_path": entry.source_path,
        "bytes_needed": entry.bytes,
        "reason": "budget",
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
