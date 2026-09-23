"""Validated resolution of operator-supplied executables and paths for the bench scripts.

The benchmark drivers take the competitor / codna binaries and fixture paths from environment
variables (``CODEX_BIN``, ``CURSOR_BIN``, ``CODNA_BIN``, ``SEC_MANIFEST``) and the bench-repos
root from argv. Those values end up as ``subprocess`` arguments, so they are validated here --
with a clear error, when a driver starts running -- instead of being passed through verbatim:

* an executable is either a bare command name from an explicit allowlist, resolved through
  ``PATH`` with :func:`shutil.which`, or an absolute path to an existing executable file;
* a file / directory override must exist and is used as an absolute path.

Nothing here ever goes through a shell, and every rejection is a ``SystemExit`` with the
variable name and the offending value so a mis-set override fails before any work starts.
The drivers call these helpers from ``main()`` / the function that spawns the subprocess, never
at module level: importing a driver must succeed on a machine without the tool installed.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Iterable, Sequence

# Provider-qualified model ids (`openai/gpt-5`, `claude-sonnet-4-6`, `gpt-5.4`): letters, digits and
# . _ : / - only, never starting with '-' so the value can never be read as an option by the tool.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def resolve_bin(env_var: str, default: str, *, allowed: Iterable[str]) -> str:
    """Absolute path of the executable named by ``$env_var`` (or ``default``).

    A bare name must be one of ``allowed`` and resolvable on ``PATH``; anything containing a
    path separator must be an absolute path (``~`` is expanded) to an existing executable file.
    """
    names = tuple(allowed)
    raw = os.environ.get(env_var) or default
    candidate = os.path.expanduser(raw)
    if os.sep in candidate:
        if not os.path.isabs(candidate):
            raise SystemExit(
                f"{env_var}={raw!r}: a path override must be absolute "
                f"(or use a bare command name from {list(names)})"
            )
        if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
            raise SystemExit(f"{env_var}={raw!r}: not an existing executable file")
        return candidate
    if candidate not in names:
        raise SystemExit(
            f"{env_var}={raw!r}: not an allowed command name {list(names)}; "
            f"pass an absolute path to the executable instead"
        )
    found = shutil.which(candidate)
    if not found:
        raise SystemExit(f"{env_var}: {candidate!r} not found on PATH")
    return found


def model_id(env_var: str, default: str = "") -> str:
    """Model identifier from ``$env_var`` (or ``default``); empty means "the tool's default"."""
    raw = os.environ.get(env_var, default)
    if raw and not _MODEL_ID_RE.match(raw):
        raise SystemExit(
            f"{env_var}={raw!r}: not a model identifier (letters, digits and . _ : / - only; "
            f"must not start with '-')"
        )
    return raw


def resolve_file(env_var: str, default: str) -> str:
    """Absolute path of the existing file named by ``$env_var`` (or ``default``)."""
    raw = os.environ.get(env_var) or default
    candidate = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isfile(candidate):
        raise SystemExit(f"{env_var}={raw!r}: file not found")
    return candidate


def bench_repos_root(argv: Sequence[str] | None = None, *, default: str = "/tmp/bench-repos") -> str:
    """Absolute path of the bench-repos directory: first CLI argument, else ``default``."""
    args = list(argv) if argv is not None else sys.argv[1:]
    raw = args[0] if args else default
    root = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(root):
        raise SystemExit(f"bench repos root {raw!r} is not a directory")
    return root
