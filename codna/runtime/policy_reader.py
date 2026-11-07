"""The environment that lets the sidecar's policy READER import `codna`.

WHY THIS IS ITS OWN MODULE. `local_stack.py` sits at the 1000-line modularity ceiling
(`cli/tests/test_modularity.py`), and this concern is independent of process supervision:
it is one question — where does `python -m codna.policy_cli` find its package — asked once.

WHAT WAS WRONG. `agent-core/vendor/cline/algenta/server/policy-document.ts` evaluates every
session by running `python -m codna.policy_cli verify --file <repo>/.codna/policy.yaml` with
`PYTHONNOUSERSITE=1` and a `PYTHONPATH` assembled from `AGENT_CORE_ROOT_DIR/cli` plus a
bounded walk up from the module for a source checkout. In a source checkout NEITHER exists:
`AGENT_CORE_ROOT_DIR` is the cline data dir (`.codna/sidecar-state/...`), and the compiled
sidecar has no `import.meta` to walk from. The reader therefore exited 1:

    /usr/local/bin/python3: Error while finding module specification for
    'codna.policy_cli' (ModuleNotFoundError: No module named 'codna')

and `applySessionPolicy` refused EVERY session with 403 `policy_verdict_unavailable` —
including a run whose repository has no policy document at all, because "no verdict" is
not "allowed". Measured on this host: a durable run parked on `codna.task.plan` for ten
activity attempts without recording a fifth ledger entry.

WHY THE CLI'S OWN MODULES ARE NOT ENOUGH. In a source checkout `codna` is importable only
through the user-site editable install (`__editable__.codna-*.pth`), which the reader
deliberately removes with `PYTHONNOUSERSITE=1`. So the package has to be named explicitly:
one PYTHONPATH entry pointing at the directory CONTAINING `codna/`.
"""
from __future__ import annotations

import os
from pathlib import Path

# The reader's own escape hatch. When an operator sets it they have said where the package
# is; this module must never disagree with them.
POLICY_PYTHONPATH_OVERRIDE = "CODNA_POLICY_PYTHONPATH"


def policy_reader_env(env: dict[str, str]) -> None:
    """Make `python -m codna.policy_cli` importable for the sidecar's policy reader.

    Mutates `env` in place: the sidecar is spawned with exactly this mapping.

    Three cases, each explicit — no guessing:

    1. An operator set `CODNA_POLICY_PYTHONPATH`. Leave everything alone; their answer wins.
    2. This is a source checkout (`codna/policy_cli.py` sits beside this package). Prepend the
       CLI root, which is what the reader itself would have discovered had it had an
       `import.meta` to walk from. An existing PYTHONPATH entry is PRESERVED, not replaced.
    3. Anything else — a wheel or packaged install, where `codna` is already in site-packages
       and the reader resolves unaided. Touch nothing: inventing a path here could shadow a
       working import, which is worse than the bug being fixed.
    """
    if env.get(POLICY_PYTHONPATH_OVERRIDE, "").strip():
        return
    package_root = Path(__file__).resolve().parent.parent
    if not (package_root / "policy_cli.py").is_file():
        return
    existing = env.get("PYTHONPATH", "").strip()
    entries = [str(package_root.parent)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
