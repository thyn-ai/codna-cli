"""Fixtures for the durable suite.

The durable extra is optional, so nothing here may import `temporalio`/`langgraph` at module scope —
doing so would turn "extra not installed" into a collection error for the whole directory and hide
the one branch the base suite is supposed to exercise.

`spike_workflows.py` is deliberately NOT named `test_*`: it holds the spike's workflow class, which
must live in an import-clean module (see its own docstring), not a test module.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# `tests.durable` is a REAL package (`cli/tests/__init__.py` + this directory's `__init__.py`) and
# that is the whole fix for the sandbox re-import. Do NOT re-introduce a synthetic `sys.modules`
# entry here.
#
# It used to live in this file: it pre-registered a `types.ModuleType("tests.durable")` so the host
# process resolved the suite. That made the host green while leaving Temporal's workflow sandbox
# broken, because the sandbox keeps its OWN `sys.modules` — it re-imports a workflow's defining
# module through `Importer`, which starts from `{}` plus `sys`/`builtins`/`__main__` and resolves
# everything else from `sys.path`. A pre-seeded host entry is invisible there. With only a namespace
# portion on the path the sandbox's `import tests.durable.spike_workflows` found whatever regular
# `tests` package won `sys.path` and raised `ModuleNotFoundError`, which surfaced as
# `RuntimeError: Failed validating workflow SpikeTaskWorkflow`.
#
# A real `__init__.py` chain is the only thing that makes host and sandbox agree, so the import is
# now identical in both.
_HERE = Path(__file__).resolve().parent
_CLI_ROOT = _HERE.parents[1]
# pytest inserts the package ROOT (`cli/`) for an `__init__.py` tree, but keep this explicit so the
# suite also works when a single module is run by path from another directory.
if str(_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLI_ROOT))


def has_durable_extra() -> bool:
    return importlib.util.find_spec("temporalio") is not None and importlib.util.find_spec("langgraph") is not None


# `spike_workflows.py` holds the spike's WORKFLOW CLASS, not tests. It must be a module-level,
# import-clean module (see its docstring), and pytest collects every `.py` under a directory it was
# pointed at — which turned a helper into a collection error the moment the extra was absent. An
# explicit hook rather than `collect_ignore`, because a directly-passed directory is still scanned.
_HELPER_MODULES = {
    "spike_workflows.py",
    "record_history.py",
    "record_workflow_history.py",
    "durable_stubs.py",
    "fakes.py",
    "workflow_specs.py",
    # The shared approval driver. Not a test module: it is imported by five scenario files, and its
    # guard lives in `test_approval_driver.py` so pytest does not collect the helper as a test file.
    "approval_driver.py",
}


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001 - pytest hook signature
    return Path(collection_path).name in _HELPER_MODULES


@pytest.fixture
def durable_extra():
    """Skip (not fail) a test that needs the extra, naming the install command.

    Skips are the *point* for the optional-extra tests: the base suite must stay green without the
    extra. Tests that prove durability behaviour live in files gated with `pytest.importorskip`, so
    running them WITH the extra is a separate, explicit gate.
    """
    if not has_durable_extra():
        pytest.skip("the `durable` extra is not installed: pip install 'codna[durable]'")
    return True


@pytest.fixture
def initial_state():
    """The serializable `TaskState` payload the spike's workflows consume, as a factory."""
    from tests.durable.spike_workflows import initial_state as make

    return make


@pytest.fixture
def state_payload() -> dict:
    """A valid state payload, deep-copied per test so a mutation cannot leak."""
    return json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "task_id": "task-1",
                "turn_seq": 0,
                "session_id": None,
                "last_turn_key": None,
                "last_event_index": 0,
                "plan_ref": None,
                "checkpoint_ref": None,
                "verify_ref": None,
                "review_verdict": None,
                "pr": None,
                "budget": {
                    "max_cost_usd": None,
                    "spent_usd": 0.0,
                    "max_tokens": None,
                    "spent_tokens": 0,
                    "unpriced_tokens": 0,
                },
                "pending_approval": None,
                "attempts": {"plan": 0, "implement": 0, "verify": 0, "review": 0, "open_pr": 0},
            }
        )
    )


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir(parents=True)
    return root
