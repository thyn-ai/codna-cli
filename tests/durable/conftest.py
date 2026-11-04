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
import types
from pathlib import Path

import pytest

# Make `tests.durable` resolve to THIS repository's tests, however pytest was invoked.
#
# The suite's modules say `from tests.durable import fakes`. Neither `cli/tests` nor `cli/tests/durable`
# carries an `__init__.py`, so `tests` is only a NAMESPACE package candidate — and Python prefers a
# REGULAR package found anywhere on the path over a namespace portion. Measured: this interpreter has
# an unrelated third-party `tests/__init__.py` in site-packages, which wins, and every module then dies
# with `ModuleNotFoundError: No module named 'tests.durable'`. It only ever worked in the dev venv
# because an editable install put `cli` on the path first and that venv had no competing `tests`.
#
# Registering `tests.durable` explicitly makes the resolution deterministic in both environments: the
# import finds it in `sys.modules` and never consults the shadowing package. Scoped to this suite — the
# parent `tests` entry is only created when nothing else provides one.
_HERE = Path(__file__).resolve().parent
_CLI_ROOT = _HERE.parents[1]
if str(_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLI_ROOT))
if "tests.durable" not in sys.modules:
    if "tests" not in sys.modules:
        parent = types.ModuleType("tests")
        parent.__path__ = [str(_HERE.parent)]
        sys.modules["tests"] = parent
    durable_pkg = types.ModuleType("tests.durable")
    durable_pkg.__path__ = [str(_HERE)]
    sys.modules["tests.durable"] = durable_pkg
    setattr(sys.modules["tests"], "durable", durable_pkg)


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
