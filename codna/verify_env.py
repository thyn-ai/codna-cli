"""The PATH a verification subprocess runs with — resolved from the user's login shell.

── THE MEASURED DEFECT ────────────────────────────────────────────────────────────────────────────

Run ``run-20261113T225225Z`` built a correct three-file app (``index.html`` 13 lines, ``app.js`` 37,
``tests/unit.test.js`` 99) and verified it three times. Every attempt failed with:

    exit 127  /bin/sh: node: command not found

The same command, in the same directory, from the user's terminal passes 3/3. The code was right and
the verdict was wrong. The cause is not path resolution inside Codna: the verify step built its
environment with ``scrub_env(dict(os.environ))``, which is a pure FILTER over the worker's own
environment. ``scrub_env`` is a security boundary — it removes write credentials — so it deliberately
leaves ``PATH`` alone, and the sandbox therefore inherited the WORKER's PATH. A worker started by a
launcher (the desktop app, launchd, a background supervisor) gets a minimal PATH such as
``/usr/bin:/bin``, while ``node``, ``bun``, ``uv`` and ``python3`` live in ``/opt/homebrew/bin`` and
friends — directories that only a LOGIN shell puts on PATH (``/etc/zprofile`` → ``path_helper``,
``~/.zprofile``'s ``brew shellenv``).

So the product assumed the ambient PATH contains the user's toolchain, which is false for every worker
that was not started from an interactive login shell. The consequence is severe and silent: correct
generated code is rejected, retries are spent on a failure that never happened, and the run reaches
``approve`` with verification that never passed.

── THE FIX, AND WHY IT HAS THIS SHAPE ────────────────────────────────────────────────────────────

Resolve the LOGIN SHELL's PATH once per process (``$SHELL -lc``, then the platform default, then the
other known shells) and merge it into the environment handed to the sandbox. Not a hardcoded
``/opt/homebrew/bin``: that is one machine's layout, wrong on Linux CI and on Intel Macs, and it would
go stale the moment a user moves their toolchain. The login shell is the only source of truth the
machine itself maintains for "where this user's tools are".

Merge order is PROCESS PATH FIRST, then login entries not already present. Three reasons:

1. **Additive, never overriding.** Verification that already worked keeps resolving exactly the same
   binaries; the change can only make a name that was NOT found become found. Putting login entries
   first would silently switch which ``node``/``python3`` runs for every existing user.
2. **Explicit injection wins.** Whoever started the worker (a venv, a CI image, the packaged app's own
   runtime) placed its directories there on purpose; the user's dotfiles must not outrank it.
3. **``/bin/sh -lc`` resolves left to right**, so the order we build is the order that decides.

``CODNA_VERIFY_PATH`` is the escape hatch and the kill switch: when set, it REPLACES the login-shell
probe (it is not prepended to it). An empty value means "do not look for a login shell at all" — verify
then runs with the worker's own PATH, which is what a hermetic CI run wants.

── CACHE, AND ITS LIFETIME ──────────────────────────────────────────────────────────────────────

The probe costs one ``zsh -lc`` (~50 ms), which is not worth paying per verify attempt, so the resolved
environment is cached for the LIFE OF THE PROCESS, keyed on ``(CODNA_VERIFY_PATH, PATH, shell
candidates, probe runner)``. That key is the whole invalidation story, and it is deliberate: a PATH only
changes when the user edits dotfiles, and a durable worker is a long-lived process whose environment the
owner cannot mutate from outside. The RUNNER is in the key because it decides the answer: a caller that
supplies a different ``run`` is asking a different question, and serving it the previous probe's answer
would silently ignore the argument the signature promises to honour. There is no TTL because a TTL would
make the same input produce different environments minutes apart — determinism beats freshness here.
Callers that know the environment moved (a reload in tests, an embedded long-runner that re-reads
config) call :func:`reset_cache`.

── FAILURE MODE ────────────────────────────────────────────────────────────────────────────────

The probe never raises and never produces an empty PATH. If no login shell answers — missing binary,
non-zero exit, timeout, missing marker, empty output — the process PATH (or ``os.defpath`` when the
worker has none either) is used and the REASON is carried in :attr:`VerifyEnvironment.note`, which the
verify step writes to the attempt's log and to the run ledger. A fallback that hides itself would turn
"we could not look" into "your tests failed", which is the defect this module exists to end.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

__all__ = [
    "EXPLICIT_ENV",
    "LoginPath",
    "PROBE_TIMEOUT_S",
    "VerifyEnvironment",
    "login_shell_path",
    "merge_paths",
    "resolve_verify_environment",
    "reset_cache",
]

#: Explicit operator override: the extra PATH entries verification gets, instead of the login shell's.
EXPLICIT_ENV = "CODNA_VERIFY_PATH"

#: One probe is bounded so a wedged dotfile cannot stall a verify activity before it even starts.
PROBE_TIMEOUT_S = 5.0

_MARK_BEGIN = "__CODNA_LOGIN_PATH_BEGIN__"
_MARK_END = "__CODNA_LOGIN_PATH_END__"
# The markers make the answer extractable even when a user's dotfiles print a banner or a prompt on
# stdout; `%s` takes "$PATH" as an ARGUMENT, so a `%` inside PATH is not reinterpreted.
_PROBE_SCRIPT = f"printf '{_MARK_BEGIN}%s{_MARK_END}' \"$PATH\""

#: Injectable for tests: the callable that runs the probe.
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class LoginPath:
    """What the login-shell probe found: the PATH, the shell that reported it, or why it did not."""

    path: str | None
    shell: str | None
    reason: str | None


@dataclass(frozen=True)
class VerifyEnvironment:
    """The environment a verification subprocess runs with, plus how its PATH was decided.

    ``env`` is a READ-ONLY view of the mapping shared with the process-lifetime cache. It is immutable
    rather than merely documented as read-only: a caller that wrote to what it received would otherwise
    corrupt the cached environment for every later attempt in the worker, and the corruption would be
    invisible until an unrelated verify resolved a PATH nobody asked for. ``dict(environment.env)`` makes
    the mutable copy anything that needs to mutate it should take.
    """

    env: Mapping[str, str]
    path: str
    #: ``"login-shell"``, ``"explicit"`` (CODNA_VERIFY_PATH), or ``"process"`` (the probe failed).
    source: str
    shell: str | None
    #: A sentence naming what was done and why — recorded so a `127` is diagnosable after the fact.
    note: str

    def describe(self) -> str:
        """One line for the verify log and ledger: where the PATH came from, and what it was.

        The PATH itself is here on purpose: an attempt that exits 127 has to say WHICH PATH it exited
        127 under, or the next reader repeats this investigation.
        """
        via = f" via {self.shell}" if self.shell else ""
        return f"{self.source}{via}: {self.note}; PATH={self.path}"


_CACHE: dict[tuple, VerifyEnvironment] = {}


def reset_cache() -> None:
    """Drop the cached environment. For tests, and for a runner that knows the user's PATH changed."""
    _CACHE.clear()


def _shell_candidates(env: Mapping[str, str]) -> tuple[str, ...]:
    """The shells to probe, deterministically ordered: the user's own first, then platform defaults.

    ``$SHELL`` is the shell whose login files actually define this user's PATH, so it is asked first.
    It must be an absolute, executable path — a login shell is only useful if it exists.
    """
    candidates: list[str] = []
    shell = (env.get("SHELL") or "").strip()
    if os.path.isabs(shell) and os.access(shell, os.X_OK):
        candidates.append(shell)
    preferred = (
        ("/bin/zsh", "/bin/bash", "/bin/sh")
        if sys.platform == "darwin"
        else ("/bin/bash", "/bin/zsh", "/bin/sh")
    )
    for candidate in preferred:
        if candidate not in candidates and os.path.exists(candidate):
            candidates.append(candidate)
    return tuple(candidates)


def login_shell_path(
    env: Mapping[str, str],
    *,
    timeout_s: float = PROBE_TIMEOUT_S,
    run: Runner = subprocess.run,
) -> LoginPath:
    """Ask the login shell for ``$PATH``. Never raises; a failure is returned as ``reason``.

    The probe's OWN environment is scrubbed with `codna.sandbox.scrub_env` regardless of what the caller
    passed: a login shell sources the user's dotfiles, and those are arbitrary code. It would be enough
    for today's caller to hand over an already-scrubbed environment, but a security property that holds
    only because of a caller's discipline is one refactor away from leaking a write token into a
    ``~/.zshrc``. Scrubbing here makes it a property of the function.
    """
    from .sandbox import scrub_env

    probe_env = {key: value for key, value in scrub_env(dict(env)).items() if isinstance(value, str)}
    candidates = _shell_candidates(probe_env)
    if not candidates:
        return LoginPath(
            None, None, "no login shell was found (looked for $SHELL, /bin/zsh, /bin/bash, /bin/sh)"
        )

    failures: list[str] = []
    for shell in candidates:
        try:
            proc = run(
                [shell, "-lc", _PROBE_SCRIPT],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=probe_env,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{shell} did not answer within {timeout_s:g}s")
            continue
        except OSError as exc:
            failures.append(f"{shell} could not be started ({exc})")
            continue
        if proc.returncode != 0:
            failures.append(f"{shell} exited {proc.returncode}")
            continue
        stdout = proc.stdout or ""
        start = stdout.find(_MARK_BEGIN)
        end = stdout.find(_MARK_END, start + len(_MARK_BEGIN)) if start >= 0 else -1
        if start < 0 or end < 0:
            failures.append(f"{shell} printed no PATH marker")
            continue
        path = stdout[start + len(_MARK_BEGIN) : end].strip()
        if not path:
            failures.append(f"{shell} reported an empty PATH")
            continue
        return LoginPath(path, shell, None)
    return LoginPath(None, None, "; ".join(failures))


def merge_paths(process_path: str, extra: str) -> str:
    """``process_path`` first, then ``extra``'s entries that are not already present, in order.

    Process entries keep their position because they are the caller's deliberate injection; duplicates
    are dropped so a login shell that already repeats a directory does not grow PATH every call.
    """
    merged: list[str] = []
    for entry in (*process_path.split(os.pathsep), *extra.split(os.pathsep)):
        if entry and entry not in merged:
            merged.append(entry)
    return os.pathsep.join(merged)


def resolve_verify_environment(
    *,
    base_env: Mapping[str, str] | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
    run: Runner = subprocess.run,
) -> VerifyEnvironment:
    """The environment verification runs with: ``base_env`` plus a PATH that reaches the user's tools.

    ``base_env`` is the worker's already-scrubbed environment (``codna.sandbox.scrub_env``); it is
    copied, never mutated. Credential filtering stays entirely where it is — this only decides PATH.
    """
    env = dict(os.environ if base_env is None else base_env)
    process_path = (env.get("PATH") or "").strip()
    explicit = env.get(EXPLICIT_ENV)
    # `run` is part of the key because it decides the answer: a caller that supplies a different probe
    # runner is asking a different question, and a cache hit on the old one would silently ignore it.
    key = (explicit, process_path, _shell_candidates(env), run)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    notes: list[str] = []
    if explicit is not None:
        extra = explicit.strip()
        source, shell = "explicit", None
        if extra:
            notes.append(f"PATH extended by {EXPLICIT_ENV}; the login-shell probe was skipped")
        else:
            notes.append(
                f"{EXPLICIT_ENV} is set but empty: verification runs with the worker's own PATH only"
            )
    else:
        found = login_shell_path(env, timeout_s=timeout_s, run=run)
        extra, shell = (found.path or ""), found.shell
        if found.path:
            source = "login-shell"
            notes.append(f"worker PATH merged with the login-shell PATH from {found.shell}")
        else:
            source = "process"
            notes.append(f"login-shell PATH unavailable ({found.reason}); using the worker's own PATH")

    if process_path:
        base = process_path
    else:
        base = os.defpath
        notes.append(f"the worker had no PATH, so {os.defpath} was used")

    path = merge_paths(base, extra)
    env["PATH"] = path
    resolved = VerifyEnvironment(
        # A read-only view of the dict, not the dict itself: a caller cannot mutate the cached
        # environment through the value it was handed.
        env=MappingProxyType(env), path=path, source=source, shell=shell, note="; ".join(notes)
    )
    _CACHE[key] = resolved
    return resolved