from __future__ import annotations

from itertools import count
import subprocess

import codna.cli as cli_module
from codna.local_client import LocalCodnaRuntimeClient


class _FakeClient:
    def __init__(self) -> None:
        self.connector_names: list[str] = []
        self.connector_configs: list[dict[str, object]] = []
        self.snapshot_requests: list[dict[str, object]] = []

    def create_connector(self, *, name: str, connector_type: str, config: dict[str, object]) -> dict[str, str]:
        self.connector_names.append(name)
        self.connector_configs.append(dict(config))
        return {"id": f"{connector_type}-1"}

    def create_repository_snapshot(self, repository_id: str, request: dict[str, object]) -> dict[str, str]:
        self.snapshot_requests.append(dict(request))
        return {"snapshot_id": f"{repository_id}-snapshot"}


def test_register_uses_collision_resistant_connector_names(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli_module.time, "time_ns", lambda: 123456789)
    monkeypatch.setattr(cli_module, "_CONNECTOR_NAME_COUNTER", count())
    client = _FakeClient()

    cli_module._register(client, str(tmp_path), None)
    cli_module._register(client, str(tmp_path), None)

    assert client.connector_names == [
        "codna-123456789-0",
        "codna-123456789-1",
    ]


def _run_git(repo, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _init_clean_repo(repo) -> str:
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "cli-test@example.com")
    _run_git(repo, "config", "user.name", "Codna CLI Test")
    (repo / "app.py").write_text("print('ok')\n", encoding="utf-8")
    _run_git(repo, "add", "app.py")
    _run_git(repo, "commit", "-m", "initial")
    return _run_git(repo, "rev-parse", "--verify", "HEAD").lower()


def test_register_marks_clean_local_git_repo_with_resolved_revision(tmp_path) -> None:
    repo = tmp_path / "repo"
    head = _init_clean_repo(repo)
    client = _FakeClient()

    cli_module._register(client, str(repo), None)

    assert client.connector_configs == [
        {
            "path": str(repo),
            "assume_clean_git_clone": True,
            "resolved_revision": head,
        }
    ]


def test_register_does_not_mark_dirty_local_git_repo_clean(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_clean_repo(repo)
    (repo / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    client = _FakeClient()

    cli_module._register(client, str(repo), None)

    assert client.connector_configs == [{"path": str(repo)}]


def test_register_passes_focus_paths_to_snapshot(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    client = _FakeClient()

    cli_module._register(client, str(repo), None, focus_paths=["src/app.py", "src/app.py"])

    assert client.snapshot_requests == [{"focus_paths": ["src/app.py"]}]


def test_issue_focus_paths_filters_to_existing_repo_files(tmp_path) -> None:
    repo = tmp_path / "repo"
    target = repo / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('ok')\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")

    paths = cli_module._issue_focus_paths(
        str(repo),
        "\n".join(
            [
                'File "src/app.py", line 1',
                f'File "{outside}", line 1',
                "Target file: missing.py",
            ]
        ),
    )

    assert paths == ["src/app.py"]


def test_client_defaults_to_in_process_local_client_without_starting_engine(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.delenv("CODNA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_BASE_URL", raising=False)
    monkeypatch.setattr(
        cli_module,
        "ensure_running",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not start HTTP engine")),
    )

    client = cli_module._client()

    assert isinstance(client, LocalCodnaRuntimeClient)
