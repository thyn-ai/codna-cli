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


#: Modules that must stay importable WITHOUT the OPTIONAL `durable` extra — the invariant
#: `tests/durable/conftest.py` states and the offline CI job depends on. The extra is not installed
#: there, so a module-scope ``from temporalio import …`` anywhere on their import chain turns
#: "extra missing" into a collection error for whoever imports them.
EXTRA_FREE_MODULES = (
    # The delegation policy read and the sentence built from it: a file read and a string builder.
    # `capability_report` used to define the sentence and imports `workflow_support` (→ temporalio),
    # which made a PURE function need the extra to be tested; the note moved here (the activity body
    # came with it) and the split is what this pins.
    "codna.durable.delegation_policy",
)


@pytest.mark.parametrize("module", EXTRA_FREE_MODULES)
def test_extra_free_modules_import_without_the_durable_extra(module: str):
    """Import ``module`` in a subprocess where `temporalio`/`langgraph` CANNOT be imported.

    A subprocess with a ``sys.meta_path`` blocker, not this process: the host has the extra installed
    here and pytest may already have imported it, so an in-process check would prove nothing. The
    blocker reproduces the offline job exactly — any module-scope import of the extra raises.
    """
    script = textwrap.dedent(
        f"""
        import importlib.abc, sys

        BLOCKED = ("temporalio", "langgraph", "langchain_core", "langsmith")

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in BLOCKED:
                    raise ModuleNotFoundError(
                        f"No module named {{fullname!r}} (blocked: the `durable` extra)", name=fullname
                    )
                return None

        sys.meta_path.insert(0, Blocker())
        import {module}
        leaked = sorted(name for name in sys.modules if name.split(".", 1)[0] in BLOCKED)
        assert not leaked, leaked
        print("ok")
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
    assert completed.returncode == 0, (
        f"{module} is not importable without the OPTIONAL `durable` extra:\n{completed.stderr}"
    )
    assert completed.stdout.strip() == "ok"


def test_the_extra_free_guard_actually_detects_the_extra():
    """The guard above must fail on a module that DOES import the extra, or it proves nothing.

    `capability_report` is the exact module the split moved the note OUT of: it schedules an activity
    through `workflow_support`, so it imports ``temporalio``. If the blocked subprocess above ever
    silently allowed that import, every assertion in it would be vacuous.
    """
    script = textwrap.dedent(
        """
        import importlib.abc, sys

        BLOCKED = ("temporalio",)

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in BLOCKED:
                    raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import codna.durable.capability_report
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
    assert completed.returncode != 0, (
        "`capability_report` imported successfully with `temporalio` blocked, so it no longer needs "
        "the extra — which would make the guard above vacuous rather than a guard"
    )
    assert "temporalio" in completed.stderr


def test_the_sandbox_canary_actually_detects_a_violation():
    """The guard itself is tested: a module that imports the forbidden layer IS reported.

    Without this, a typo in FORBIDDEN (or a `_imported_modules` that silently returned an empty
    set) would make every assertion above vacuously true.
    """
    pulled = _imported_modules("codna.runtime.config")  # imports httpx by design
    assert "urllib.request" in pulled


def test_the_workflow_module_is_resolvable_inside_the_temporal_sandbox():
    # The `cli secure subsystem` lane is deliberately dependency-minimal (no `temporalio`), and this
    # assertion is *about* the sandbox importer, so it cannot run without it. Skip cleanly rather than
    # erroring: an ImportError here is a statement about the runner, not about the package layout this
    # test exists to protect. The `cli durable` lane (which installs `codna[durable]`) is where this
    # test actually carries weight.
    pytest.importorskip("temporalio", reason="the `durable` extra is not installed on this lane")
    """The sandbox must re-import `tests.durable.spike_workflows` from `sys.path`, unaided.

    This is the regression net for the failure that was blamed on sandbox flakiness for the life of
    the project:

        ModuleNotFoundError: No module named 'tests.durable'
        RuntimeError: Failed validating workflow SpikeTaskWorkflow

    Temporal's sandbox does not share the host `sys.modules`. It builds its own (`sys`, `builtins`,
    `__main__`) and resolves everything else from `sys.path`, so any host-side pre-registration of
    `tests.durable` is invisible to it. When `cli/tests` had no `__init__.py` it was only a
    NAMESPACE portion, and a namespace portion loses to a regular `tests` package earlier on the
    path — on this machine an unrelated `tests/__init__.py` in user site-packages. The sandbox then
    found that one and the import failed.

    The fix is the `__init__.py` chain that makes `tests` a real package, so host and sandbox agree.
    This test asserts the resolution FROM INSIDE the sandbox importer, which is the only place the
    bug was ever visible. It is deliberately not "the workflow validated": with passthrough
    restrictions absent that can fail for a different, unrelated reason (the plugin supplies them).
    """
    from temporalio.worker.workflow_sandbox import SandboxRestrictions
    from temporalio.worker.workflow_sandbox._importer import Importer
    from temporalio.worker.workflow_sandbox._restrictions import RestrictionContext

    # The restrictions the LangGraph plugin installs (`temporalio.contrib.langgraph`). Without them the
    # import fails for an UNRELATED reason — `langgraph` is sandboxed and reaches `http.client` — and
    # this test would be measuring the plugin's job rather than the package layout.
    restrictions = SandboxRestrictions.default.with_passthrough_modules(
        "langchain", "langchain_core", "langgraph", "langsmith", "numpy"
    )
    importer = Importer(restrictions, RestrictionContext())
    with importer.applied():
        import builtins

        # `fromlist` so the LEAF module is returned, not the top-level package.
        module = builtins.__import__(
            "tests.durable.spike_workflows", fromlist=["spike_workflows"]
        )
        resolved = module.__file__

    assert Path(resolved).resolve() == Path(__file__).resolve().parent / "spike_workflows.py", (
        "the sandbox resolved `tests.durable.spike_workflows` to somewhere other than this suite: "
        f"{resolved}. `cli/tests` and `cli/tests/durable` must each keep an `__init__.py`, or a "
        "regular `tests` package elsewhere on sys.path wins and every workflow validation fails."
    )

