from __future__ import annotations

from codna import cli, scaffold, status


def test_init_writes_files(tmp_path):
    res = scaffold.init_project(tmp_path)
    assert res == {"codna.yaml": "written", "AGENTS.md": "written"}
    config = (tmp_path / "codna.yaml").read_text()
    assert "# model:" in config
    assert "#   provider: openai" in config
    assert "#   key: env:CODNA_MODEL_KEY" in config
    assert "privacy:" in config
    assert "  redact_secrets: true" in config
    assert "# egress: fail-closed" in config
    assert "  egress: fail-closed" not in config
    assert (tmp_path / "AGENTS.md").is_file()


def test_init_no_clobber_without_force(tmp_path):
    (tmp_path / "codna.yaml").write_text("KEEP ME\n", encoding="utf-8")
    res = scaffold.init_project(tmp_path)
    assert res["codna.yaml"] == "exists"
    assert (tmp_path / "codna.yaml").read_text() == "KEEP ME\n"  # untouched
    # force overwrites
    assert scaffold.init_project(tmp_path, force=True)["codna.yaml"] == "written"


def test_init_no_agents(tmp_path):
    res = scaffold.init_project(tmp_path, with_agents=False)
    assert "AGENTS.md" not in res
    assert not (tmp_path / "AGENTS.md").exists()


def test_status_lines_degrade_gracefully(monkeypatch):
    # No engine, no telys, no keys — status must still produce lines, never raise.
    for name in ("CODNA_API_KEY", "ALGENTA_API_KEY", "DE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    lines = status.build_status_lines({}, api_key_present=False)
    joined = "\n".join(lines)
    assert joined.startswith("codna status:")
    assert "engine" in joined and "telys runtime" in joined and "provider key" in joined


def test_parser_has_init_status_and_mcp_install():
    p = cli.build_parser()
    assert p.parse_args(["init", "--force"]).force is True
    assert p.parse_args(["status"]).func is cli.cmd_status
    a = p.parse_args(["mcp", "install", "--client", "cursor"])
    assert a.action == "install" and a.client == "cursor"


def test_cmd_mcp_install_requires_client(capsys):
    from types import SimpleNamespace
    from codna.cli import CodnaError
    import pytest
    with pytest.raises(CodnaError):
        cli.cmd_mcp(SimpleNamespace(action="install", client=None, repo=None, project=False))


def test_cmd_status_does_not_read_keychain(monkeypatch, capsys):
    from codna import keystore

    def forbidden_keychain_read():
        raise AssertionError("status must not read OS keychain secrets")

    monkeypatch.setattr(keystore, "config_values", forbidden_keychain_read)

    rc = cli.cmd_status(object())

    assert rc == 0
    assert "codna status:" in capsys.readouterr().out
