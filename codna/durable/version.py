"""`GRAPH_VERSION` and the build id — a LEAF module, and that is the whole point.

This exists separately from `config.py` because of a D2.0 spike finding that would otherwise be a
production-only failure:

    RestrictedWorkflowAccessError: Cannot access urllib.request.Request.__mro_entries__
    from inside a workflow → RuntimeError: Failed validating workflow <T>

Temporal re-imports a workflow's defining module inside the workflow sandbox, and that sandbox
refuses `urllib.request`. `config.py` needs `codna.runtime.config` — which imports `httpx`, which
imports `urllib.request` — so **anything a workflow module imports must not reach `config.py`**.
`graph.py` and the future `workflows.py` need only the version, so the version lives here: stdlib
and the package version, nothing else.

Verified by `tests/durable/test_import_hygiene.py`, which asserts that importing `graph` does not
pull `urllib.request` (or `httpx`, or `subprocess`) into `sys.modules`. That test is the guard that
keeps this split from being undone by a harmless-looking import.
"""
from __future__ import annotations

from typing import Final

from .. import __version__

__all__ = ["GRAPH_VERSION", "build_id_for"]

# The version of the task graph itself, independent of the CLI version. Bumping this is the
# documented way to change the graph: it changes `build_id`, which pins a new worker version, and it
# is the signal that the replay histories in cli/tests/durable/histories/ must be re-recorded.
GRAPH_VERSION: Final = 1


def build_id_for(version: str = __version__, graph_version: int = GRAPH_VERSION) -> str:
    """``codna-<ver>-g<GRAPH_VERSION>`` — the worker build id, and therefore the pinning key.

    Deterministic and derived only from things in the code, so two workers from the same checkout
    always agree, and a graph change always produces a different one.
    """
    return f"codna-{version}-g{graph_version}"
