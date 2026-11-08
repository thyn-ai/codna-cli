"""`codna init` — scaffold a codna.yaml (and an optional AGENTS.md stub) in one command.

Onboarding: writes a minimal, commented codna.yaml (model + privacy) the loader understands, without
clobbering an existing file (use --force to overwrite). Prints what it created + next steps.
"""
from __future__ import annotations

from pathlib import Path

_CODNA_YAML = """\
# codna configuration. See `codna --help`. Privacy applies broadly.
#
# Optional model config for `codna fix`.
# Uncomment only after setting the referenced environment variable; otherwise
# Codna fails closed rather than silently guessing a provider key.
# model:
#   provider: openai            # openai | anthropic | gemini | google | groq | mistral | openrouter | xai
#   key: env:CODNA_MODEL_KEY    # env:NAME reads $NAME; keeps the secret out of the file
#
# Optional: how `codna fix --tests` (and the GitHub App) runs this repository's tests.
# Default is pytest, or the runner the repo declares (a pixi `test` task, uv.lock).
# fix:
#   test_command: pixi run test
privacy:
  redact_secrets: true        # posture; codna never disables secret redaction
  # Optional stricter sandbox for environments with kernel-level network isolation
  # such as bwrap/unshare. If unavailable, Codna refuses to run tests open.
  # egress: fail-closed
"""

_AGENTS_MD = """\
# AGENTS.md — project guidance for Codna's fix agent

- Test command: `pytest -q`      # how to run the suite (used by `codna fix --tests`)
- Style: match surrounding code; no unrelated refactors.
- Do not touch: build artifacts, vendored code, generated files.
"""

# The `.codnaignore` template. Deliberately ALL comments: a scaffolded ignore file that
# pre-excluded something would be a behaviour change the user did not ask for, and an empty
# rule set is what "no opinion yet" looks like. The format note is what makes the file
# self-explanatory when a user opens it to add a rule.
_CODNAIGNORE = """\
# .codnaignore — paths Codna must not index, recall, or read.
#
# Same syntax as .gitignore: `*` (not across /), `**` (any depth), `?`, `[abc]`, `#` comments,
# trailing `/` for directories only, `/pattern` anchored to this directory, `!pattern` to negate.
# The LAST matching pattern wins. Files in a subdirectory apply only to that subtree.
#
# Precedence, lowest to highest: .gitignore -> .cursorignore -> .codnaignore. Your .gitignore and
# an existing .cursorignore are read automatically, so a Cursor repo needs no changes.
#
# A built-in floor (secrets, VCS internals, node_modules, build output, hidden directories) is
# always applied and cannot be weakened here: `!.env` re-exposes nothing.
#
# Examples:
#   data/            # a directory the index must skip
#   fixtures/        # test fixtures you never want embedded
#   *.generated.py   # a generated-file pattern
#   !docs/**         # re-include something an earlier rule excluded
"""


class ScaffoldError(Exception):
    """Surfaced to the user."""


def _write(path: Path, content: str, *, force: bool) -> str:
    if path.exists() and not force:
        return "exists"
    path.write_text(content, encoding="utf-8")
    return "written"


def init_project(cwd: Path | None = None, *, force: bool = False, with_agents: bool = True) -> dict:
    root = cwd or Path.cwd()
    result = {"codna.yaml": _write(root / "codna.yaml", _CODNA_YAML, force=force)}
    if with_agents:
        result["AGENTS.md"] = _write(root / "AGENTS.md", _AGENTS_MD, force=force)
    # The ignore template is always scaffolded: it is the one file that decides what the index
    # sees, and a user who never learns it exists keeps paying for embeddings they excluded.
    result[".codnaignore"] = _write(root / ".codnaignore", _CODNAIGNORE, force=force)
    return result
