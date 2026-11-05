"""Keep the workflow-facing modules importable inside Temporal's workflow sandbox.

Temporal re-imports a workflow's defining module inside a restricted sandbox, and that sandbox
refuses `urllib.request`, `http.client`, `subprocess` and friends. A module that a workflow imports
transitively and that reaches any of those fails at WORKER CONSTRUCTION with

    RestrictedWorkflowAccessError: Cannot access urllib.request.Request.__mro_entries__
    from inside a workflow → RuntimeError: Failed validating workflow <T>

The D2.0 spike hit exactly this: `graph.py` originally imported `GRAPH_VERSION` from `config.py`,
which needs `codna.runtime.config`, which imports `httpx`, which imports `urllib.request`. No test
that ran a workflow could pass. The fix was `version.py` — a leaf module with no codna imports
beyond the package version — and this file is the guard that stops the split from being undone by a
harmless-looking import in a later commit.

`subprocess` is NOT asserted absent: `temporalio` itself imports it and the sandbox passes it
through. The assertion is about what *we* pull in, which is why `urllib.request`/`http.client`/
`httpx` are the canaries — they are only reachable through our own HTTP/runtime layers.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CLI_DIR = Path(__file__).resolve().parents[2]

# Modules that must not be reachable from a workflow-facing module. Each is only pulled in by
# codna's own runtime/HTTP layers (or by httpx), so their presence means a workflow module started
# depending on something it will not have inside the sandbox.
FORBIDDEN = ("urllib.request", "http.client", "httpx", "socket")


def _imported_modules(module: str) -> set[str]:
    """Import ``module`` in a FRESH interpreter and report the modules it pulled in.

    A subprocess, not an in-process import: pytest has already imported httpx (its own machinery and
    other test modules do), so an in-process check would always fail for reasons unrelated to the
    module under test.
    """
    script = textwrap.dedent(
        f"""
        import json, sys
        before = set(sys.modules)
        import {module}
        print(json.dumps(sorted(set(sys.modules) - before)))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(CLI_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    import json

    return set(json.loads(completed.stdout))


@pytest.mark.parametrize(
    "module",
    [
        "codna.durable.graph",
        "codna.durable.version",
        "codna.durable.routing",
        "codna.durable.state",
        # `provider.py` was split out of `contracts.py` for the 1000-line ceiling, and it is a leaf the
        # graph and the plain workflow both reach. If it ever gained a keystore import (the obvious
        # place to "just look up the key"), it would drag `codna.runtime.config` → httpx into the
        # sandbox and fail the worker at construction — so the guard covers it explicitly.
        "codna.durable.provider",
        "codna.durable.contracts",
    ],
)
def test_workflow_facing_modules_stay_sandbox_clean(module: str):
    pulled = _imported_modules(module)
    offending = [name for name in FORBIDDEN if name in pulled]
    assert not offending, (
        f"{module} pulls {offending} into sys.modules, which Temporal's workflow sandbox refuses. "
        "Move the shared constant/helper into a leaf module (see codna/durable/version.py)."
    )


def test_version_is_a_true_leaf():
    """`version.py` must import nothing from codna but the package version itself."""
    source = (CLI_DIR / "codna" / "durable" / "version.py").read_text(encoding="utf-8")
    codna_imports = [
        line.strip()
        for line in source.splitlines()
        if line.startswith("from codna")
        or line.startswith("import codna")
        or line.lstrip().startswith("from ..")
        or line.lstrip().startswith("from .")
    ]
    assert codna_imports == ["from .. import __version__"], codna_imports


def test_graph_does_not_import_config():
    """The specific regression this exists for: `graph.py` must not import `config`."""
    source = (CLI_DIR / "codna" / "durable" / "graph.py").read_text(encoding="utf-8")
    assert "from .config import" not in source
    assert "import config" not in source
    assert "from .version import" in source


def test_the_sandbox_canary_actually_detects_a_violation():
    """The guard itself is tested: a module that imports the forbidden layer IS reported.

    Without this, a typo in FORBIDDEN (or a `_imported_modules` that silently returned an empty
    set) would make every assertion above vacuously true.
    """
    pulled = _imported_modules("codna.runtime.config")  # imports httpx by design
    assert "urllib.request" in pulled
