"""The two ENTRANCES for the worktree flag: `codna run` and `POST /runs` (D5).

Both surfaces build the same `RunSpec` through `run_payload_for`, so what is worth testing is exactly
the part each one can get wrong on its own:

* the CLI's spelling — ``--worktree`` / ``--no-worktree`` / neither, including the conflict;
* the Run API's closed field set — the field is accepted as a BOOLEAN, refused as anything else, and a
  stray field is still a 400 (the set did not widen by accident);
* and, for both, that the value reaches the payload the workflow decodes rather than stopping at the
  banner.

The `provider` coordinates are the pattern this follows (`_START_FIELDS`, a closed set, validate, 400 on
unknown), so the assertions mirror `test_run_api.py`'s provider section deliberately.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest

# The `durable` extra is OPT-IN and the offline CI job installs everything EXCEPT it. Two tests here call
# `from codna.durable.workflow_support import ...` INSIDE the test body, and that module imports
# `temporalio` at module scope — so the failure surfaced at CALL time as a ModuleNotFoundError pointing at
# `workflow_support.py:24`, not as a collection error, which is why a top-of-file guard had been missed.
# `importorskip` at module scope turns those two tests into skips in the job that lacks the extra, and
# leaves them running everywhere the extra is installed.
pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

# `durable.cli` BEFORE `durable.run_cli`: the two import each other (`run_cli` reaches `cli` for its
# shared helpers, `cli` reaches `run_cli` for the command implementations), and importing `run_cli`
# first leaves `cli` a partially-initialised module. The rest of this suite imports them in this order
# for the same reason, so following it here keeps one convention rather than adding a second.
from codna.durable import cli as durable_cli  # noqa: F401  - imported for its side effect (ordering)
from codna.durable import run_cli

# -- the CLI spelling ----------------------------------------------------------------------


def _args(**overrides: Any) -> argparse.Namespace:
    base = {"worktree": False, "no_worktree": False}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_default_is_the_shared_checkout():
    """``--no-worktree`` is the DEFAULT: a run that was not asked for checkouts creates none.

    Deliberate, and the reason is disk: a worktree is a full checkout per task, so imposing it on an
    existing caller pointed at a large repository would be a surprise the flag was meant to offer.
    """
    assert run_cli._worktree_choice(_args()) is False


def test_worktree_turns_it_on():
    assert run_cli._worktree_choice(_args(worktree=True)) is True


def test_no_worktree_turns_it_off():
    assert run_cli._worktree_choice(_args(no_worktree=True)) is False


def test_both_flags_at_once_is_a_refusal():
    """Not "last one wins": a script that passes both does not know which tree it will edit."""
    with pytest.raises(Exception) as caught:
        run_cli._worktree_choice(_args(worktree=True, no_worktree=True))
    assert getattr(caught.value, "code", "") == "durable_worktree_conflict"
    assert caught.value.details["flags"] == ["--worktree", "--no-worktree"]


def test_the_flag_exists_on_both_surfaces():
    """`codna run` and `codna durable run` share `add_run_flags`, so a flag cannot exist on one only."""
    from codna import cli as top_cli

    for argv in (["run", "--brief", "b"], ["durable", "run", "--brief", "b"]):
        parsed = top_cli.build_parser().parse_args(argv + ["--worktree"])
        assert parsed.worktree is True
        parsed = top_cli.build_parser().parse_args(argv + ["--no-worktree"])
        assert parsed.no_worktree is True


def test_the_flags_are_documented_by_their_own_help_text():
    """The help is where a user learns what `--no-worktree` MEANS, so it has to say it."""
    from codna import cli as top_cli

    # The SUBcommand's help, not the top-level one: `--worktree` belongs to `run`'s parser, and the
    # top-level parser only lists the subcommand names. `_SubParsersAction.choices` is how the parser
    # it actually built is reached, so this reads the help a user gets from `codna run --help`.
    actions = [
        action
        for action in top_cli.build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)  # noqa: SLF001 - stdlib, and the only route
    ]
    assert actions, "the CLI has no subcommands; this test would be vacuous"
    run_parser = actions[0].choices["run"]
    help_text = run_parser.format_help()
    assert "--worktree" in help_text and "--no-worktree" in help_text
    # Both spellings document the MEANING, not just their existence: "--no-worktree" is the default, so
    # a user reading the help has to be able to tell what it does.
    assert "shared checkout" in help_text


# -- the payload ---------------------------------------------------------------------------


def _config(tmp_path: Path) -> Any:
    from codna.durable.config import load_durable_config

    root = tmp_path / "runtime"
    os.environ["CODNA_RUNTIME_ROOT"] = str(root)
    env = dict(os.environ)
    env["CODNA_RUNTIME_ROOT"] = str(root)
    env["CODNA_DURABLE_RUNS_ROOT"] = str(tmp_path / "runs")
    return load_durable_config(env=env, hostname="worktree-surface-host")


def _git_repo(path: Path) -> Path:
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "test@codna.ai"],
        ["git", "config", "user.name", "test"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "init"],
    ):
        completed = subprocess.run(
            [argv[0], "-C", str(path), *argv[1:]], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
    return path


def _payload(tmp_path: Path, *, worktree: bool) -> dict[str, Any]:
    repo = _git_repo(tmp_path / "repo")
    return run_cli.run_payload_for(
        config=_config(tmp_path),
        run_id="run-wt",
        repo_dir=repo,
        briefs=["fix a", "fix b"],
        base_branch="main",
        verify_commands=("true",),
        max_cost_usd=None,
        max_tokens=None,
        max_attempts=2,
        engine="plain",
        worktree=worktree,
    )


def test_the_payload_carries_the_flag_on_the_run_and_on_every_task(tmp_path: Path):
    """Both halves are load-bearing: the RUN level decides the sweep, the TASK level the checkout."""
    payload = _payload(tmp_path, worktree=True)

    assert payload["worktree"] is True
    assert [task["worktree"] for task in payload["tasks"]] == [True, True]
    # The task's repo_dir stays the SHARED checkout — the worktree is derived from it, not substituted
    # into it, so a spec still names the repository its refs and objects belong to.
    assert all(task["repo_dir"] == payload["repo_dir"] for task in payload["tasks"])


def test_the_payload_defaults_to_the_shared_checkout(tmp_path: Path):
    payload = _payload(tmp_path, worktree=False)
    assert payload["worktree"] is False
    assert [task["worktree"] for task in payload["tasks"]] == [False, False]


def test_the_payload_survives_the_workers_own_decoder(tmp_path: Path):
    """What reaches the workflow is a SPEC a worker can decode — asserted with the worker's parser."""
    from codna.durable.workflow_support import run_spec_from

    payload = _payload(tmp_path, worktree=True)
    decoded = run_spec_from(payload)

    assert decoded.worktree is True
    assert all(task.worktree for task in decoded.tasks)
    # ...and the context a step uses carries it too, which is what the activities read.
    from codna.durable.contracts import TaskContext

    assert TaskContext.from_spec(decoded.tasks[0]).worktree is True


def test_a_payload_recorded_before_the_flag_decodes_to_the_shared_checkout(tmp_path: Path):
    """BACKWARD COMPATIBLE: an in-flight run's spec has no field, and must keep its old behaviour."""
    from codna.durable.workflow_support import run_spec_from, spec_from

    payload = _payload(tmp_path, worktree=True)
    del payload["worktree"]
    for task in payload["tasks"]:
        del task["worktree"]

    decoded = run_spec_from(payload)
    assert decoded.worktree is False
    assert spec_from(payload["tasks"][0]).worktree is False


# -- the Run API ---------------------------------------------------------------------------


def test_the_route_accepts_the_worktree_field_and_nothing_else_new():
    """The closed set widened by exactly one name, and it is a boolean the route validates."""
    from codna.durable.run_api_http import RunApiHandler

    assert "worktree" in RunApiHandler._START_FIELDS
    # No credential-shaped name crept in alongside it.
    assert not {
        name
        for name in RunApiHandler._START_FIELDS
        if name.endswith(("_key", "_key_source", "_api_key")) or name in ("api_key", "authorization")
    }


@pytest.mark.parametrize("value", ["true", "yes", 1, 0, [], {}, None])
def test_optional_bool_refuses_a_non_boolean(value: Any):
    """A coerced flag is a flag whose effect cannot be predicted from the request."""
    from codna.durable.run_api_support import RunApiError, optional_bool

    body: Mapping[str, Any] = {} if value is None else {"worktree": value}
    if value is None:
        # ABSENT is the default, not a refusal: `None` as a VALUE is a different thing.
        assert optional_bool(body, "worktree", default=False) is False
        assert optional_bool(body, "worktree", default=True) is True
        return
    with pytest.raises(RunApiError) as caught:
        optional_bool(body, "worktree", default=False)
    assert caught.value.code == "run_api_field_invalid"
    assert caught.value.status == 400


@pytest.mark.parametrize("value", [True, False])
def test_optional_bool_accepts_a_boolean(value: bool):
    from codna.durable.run_api_support import optional_bool

    assert optional_bool({"worktree": value}, "worktree", default=not value) is value


def _server(tmp_path: Path) -> Iterator[Any]:
    from codna.durable.run_api import RunApiServer
    from codna.durable.run_api_support import RunApiSettings
    from codna.durable.run_api_backend import RunBackend

    class RecordingBackend(RunBackend):
        def __init__(self) -> None:
            self.started: list[dict[str, Any]] = []

        def capabilities(self) -> Any:
            from codna.durable.run_api_backend import durable_capabilities

            return durable_capabilities("127.0.0.1:7233")

        def start_run(self, *, payload: Any, run_id: str, run_timeout_s: int) -> dict[str, Any]:
            self.started.append({"payload": dict(payload), "run_id": run_id})
            return {"run_id": run_id, "workflow_id": f"run:{run_id}", "tasks": 1}

        def status(self, run_id: str) -> dict[str, Any] | None:
            return None

        def update(self, run_id: str, *, task_id: str, action: str, payload: Any) -> Any:
            raise NotImplementedError

        def signal(self, run_id: str, action: str) -> None:
            raise NotImplementedError

        def close(self) -> None:
            return None

    config = _config(tmp_path)
    backend = RecordingBackend()
    settings = RunApiSettings(
        port=0,
        token_path=config.paths.root / "run-api" / "run-api.token",
        heartbeat_s=0.15,
        poll_interval_s=0.02,
        request_timeout_s=5.0,
        write_timeout_s=2.0,
        allowed_repo_roots=(tmp_path,),
    )
    running = RunApiServer(config=config, backend=backend, settings=settings).start()
    try:
        yield running, backend, config
    finally:
        running.stop()


def _post(server: Any, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    import http.client
    import json

    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    try:
        connection.request(
            "POST",
            "/runs",
            body=json.dumps(body),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {server.token}",
            },
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


def test_post_runs_carries_worktree_into_the_started_payload(tmp_path: Path):
    for running, backend, _config_ in _server(tmp_path):
        repo = _git_repo(tmp_path / "api-repo")
        status, body = _post(
            running,
            {
                "run_id": "run-api-wt",
                "repo_dir": str(repo),
                "briefs": ["fix the retry"],
                "engine": "plain",
                "worktree": True,
            },
        )
        assert status == 202, body
        payload = backend.started[0]["payload"]
        assert payload["worktree"] is True
        assert payload["tasks"][0]["worktree"] is True


def test_post_runs_defaults_worktree_to_false(tmp_path: Path):
    for running, backend, _config_ in _server(tmp_path):
        repo = _git_repo(tmp_path / "api-default-repo")
        status, body = _post(
            running,
            {"run_id": "run-api-default", "repo_dir": str(repo), "briefs": ["b"], "engine": "plain"},
        )
        assert status == 202, body
        assert backend.started[0]["payload"]["worktree"] is False


def test_post_runs_refuses_a_non_boolean_worktree_before_starting_anything(tmp_path: Path):
    for running, backend, _config_ in _server(tmp_path):
        repo = _git_repo(tmp_path / "api-bad-repo")
        for value in ("true", 1, "yes"):
            status, body = _post(
                running,
                {
                    "run_id": "run-api-bad",
                    "repo_dir": str(repo),
                    "briefs": ["b"],
                    "worktree": value,
                },
            )
            assert status == 400, (value, body)
            assert body["error"]["code"] == "run_api_field_invalid"
            assert body["error"]["details"]["field"] == "worktree"
        assert backend.started == []
