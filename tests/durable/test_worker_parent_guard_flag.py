"""`--guard-parent-pid` is a SUPERVISION CONTRACT, not a flag to ignore.

The desktop passes the pid of the process that forked the worker. Two properties are asserted here:

  1. THE VALUE MUST NAME THE ACTUAL PARENT. A worker whose parent changed between spawn and startup must
     refuse rather than guard an unrelated pid — `True` without a verified value is not a contract.
  2. A HAND-RUN WORKER KEEPS ITS OLD BEHAVIOUR. No flag means no guard, and a terminal losing its shell is
     not a reason to abandon an in-flight paid turn. The guard is therefore opt-in through the flag, not
     through ambient state.

Both are exercised at `cmd_worker`'s boundary by substituting the one function that builds the settings,
so no Temporal server, sidecar or real worker is started.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

import codna.durable.cli  # noqa: F401 — importing the durable package before `run_cli` resolves its re-export cycle
import codna.durable.worker as durable_worker_module
from codna.durable import run_cli
from codna.durable.errors import DurableCliError


def _args(**overrides: Any) -> SimpleNamespace:
    base = dict(
        max_concurrent_activities=2,
        no_sidecar=True,
        grace_period=10.0,
        no_promote=False,
        start_dev_server=False,
        dev_server=False,
        start_timeout=60.0,
        engine=None,
        guard_parent_pid=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def no_durable_required(monkeypatch: pytest.MonkeyPatch) -> None:
    # `cmd_worker` reaches `require_durable` through a local `from . import require_durable`, so patch
    # the defining module (`codna.durable.__init__`), not the importer.
    import codna.durable as durable_init

    monkeypatch.setattr(durable_init, "require_durable", lambda: None)


@pytest.fixture
def recorded_settings(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    real_settings = durable_worker_module.worker_settings_from_config

    def fake_settings(config: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        kwargs.setdefault("ensure_sidecar", False)
        return real_settings(config, **kwargs)

    # `cmd_worker` imports these names locally inside the function, so patch the defining module.
    monkeypatch.setattr(durable_worker_module, "worker_settings_from_config", fake_settings)
    monkeypatch.setattr(durable_worker_module, "run_worker_sync", lambda settings: 0)
    monkeypatch.setattr(run_cli._cli, "_config", lambda args: SimpleNamespace(
        build_id="codna-0.0.0-test-g1",
        address="127.0.0.1:7233",
        namespace="codna",
        task_queue="codna.agent.test",
        graph_engine="plain",
        runs_root="/tmp/runs",
        sidecar_url="http://127.0.0.1:18601",
        profile="local",
        api_key_env=None,
        pool=None,
        connect_options=lambda: {},
    ))
    monkeypatch.setattr(run_cli._cli, "_print", lambda payload: None)
    return captured


def test_a_guard_pid_that_is_not_the_actual_parent_is_refused(
    no_durable_required: None, recorded_settings: dict[str, Any]
) -> None:
    # A pid above any real process table entry, so the mismatch is deterministic and never signals anyone.
    wrong_pid = 1 << 22
    with pytest.raises(DurableCliError) as excinfo:
        run_cli.cmd_worker(_args(guard_parent_pid=wrong_pid))
    assert excinfo.value.code == "worker_parent_pid_mismatch"
    assert "does not name this worker's parent" in str(excinfo.value)


def test_a_non_integer_or_nonpositive_guard_pid_is_refused(
    no_durable_required: None, recorded_settings: dict[str, Any]
) -> None:
    for bad in (0, -1, True):
        with pytest.raises(DurableCliError) as excinfo:
            run_cli.cmd_worker(_args(guard_parent_pid=bad))
        assert excinfo.value.code == "worker_parent_pid_invalid"


def test_the_actual_parent_pid_is_accepted_and_marked_on_the_settings(
    no_durable_required: None, recorded_settings: dict[str, Any]
) -> None:
    run_cli.cmd_worker(_args(guard_parent_pid=os.getppid()))
    assert recorded_settings["guard_parent"] is True
    assert recorded_settings["guard_parent_pid"] == os.getppid()
    # An app-owned worker never promotes itself to the deployment's current version by default.
    assert recorded_settings["promote_current"] is False


def test_no_guard_flag_leaves_a_hand_run_worker_unpromoted_by_default_and_unguarded(
    no_durable_required: None, recorded_settings: dict[str, Any]
) -> None:
    run_cli.cmd_worker(_args())
    assert recorded_settings["guard_parent"] is False
    assert recorded_settings["guard_parent_pid"] is None
    # The DOCUMENTED default for a human-run worker is unchanged: it promotes a first build.
    assert recorded_settings["promote_current"] is True


def test_the_guard_is_armed_with_the_VALIDATED_pid_not_a_re_read(
    no_durable_required: None, recorded_settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pid the CLI validated is the pid the guard watches. A re-read is a TOCTOU window.

    `_validate_guard_parent_pid` proves `--guard-parent-pid` names the worker's real parent, and then
    `install_parent_guard` used to ask the OS again. Between those two reads the answer can change, and
    the guard would then be watching a process nobody checked — the defect this asserts against.
    """
    import codna.durable.parent_guard as parent_guard_module

    validated = os.getppid()
    armed: dict[str, Any] = {}

    class RecordingGuard:
        started_ppid = validated

        def stop(self) -> None:
            armed["stopped"] = True

    def fake_install(on_lost: Any, *, poll_ms: int = 0, parent_pid: int | None = None) -> Any:
        armed["on_lost"] = on_lost
        armed["parent_pid"] = parent_pid
        return RecordingGuard()

    monkeypatch.setattr(parent_guard_module, "install_parent_guard", fake_install)

    # The FIRST read is the validation (it must answer with the real parent); every later read drifts,
    # which is exactly the window: a guard that re-read the ppid would watch this other value.
    real_ppid = os.getppid()
    reads = {"count": 0}

    def drifting_ppid() -> int:
        reads["count"] += 1
        return real_ppid if reads["count"] == 1 else 1 << 23

    monkeypatch.setattr(os, "getppid", drifting_ppid)
    run_cli.cmd_worker(_args(guard_parent_pid=real_ppid))

    assert reads["count"] == 1, "the ppid was read again after validation"
    assert armed["parent_pid"] == real_ppid, "the guard was armed with a re-read pid, not the validated one"
    # Armed BEFORE the worker ran, and disarmed after it: the guard covers startup as well as the run.
    assert armed.get("stopped") is True
