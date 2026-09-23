from __future__ import annotations

from codna import cli


def test_installed_wheel_does_not_look_for_keys_txt(monkeypatch):
    """End users: an installed package must never hunt for keys.txt (no path, no stat)."""
    monkeypatch.delenv("CODNA_KEYS_FILE", raising=False)
    monkeypatch.setattr(cli, "__file__", "/venv/lib/python3.12/site-packages/codna/cli.py")
    assert cli._keys_file_path() is None


def test_dist_packages_layout_also_skipped(monkeypatch):
    monkeypatch.delenv("CODNA_KEYS_FILE", raising=False)
    monkeypatch.setattr(cli, "__file__", "/usr/lib/python3/dist-packages/codna/cli.py")
    assert cli._keys_file_path() is None


def test_source_checkout_uses_repo_keys_txt(monkeypatch, tmp_path):
    """Dev checkout keeps the repo-root keys.txt convenience."""
    monkeypatch.delenv("CODNA_KEYS_FILE", raising=False)
    fake = tmp_path / "codna-repo" / "cli" / "codna" / "cli.py"
    monkeypatch.setattr(cli, "__file__", str(fake))
    assert cli._keys_file_path() == tmp_path / "codna-repo" / "keys.txt"


def test_codna_keys_file_override_wins_even_when_installed(monkeypatch, tmp_path):
    custom = tmp_path / "custom-keys.txt"
    monkeypatch.setenv("CODNA_KEYS_FILE", str(custom))
    monkeypatch.setattr(cli, "__file__", "/venv/lib/python3.12/site-packages/codna/cli.py")
    assert cli._keys_file_path() == custom


def test_runtime_keys_ignores_keys_txt_when_installed(monkeypatch, tmp_path):
    """Even if a keys.txt sits where the installed layout would compute, it is never read."""
    monkeypatch.delenv("CODNA_KEYS_FILE", raising=False)
    monkeypatch.setattr(cli, "__file__", "/venv/lib/python3.12/site-packages/codna/cli.py")
    # keychain contributes nothing here
    from codna import keystore
    monkeypatch.setattr(keystore, "config_values", dict)
    assert cli._runtime_keys() == {}


def test_runtime_keys_does_not_probe_keychain_without_local_index(monkeypatch, tmp_path):
    """A clean user install must not trigger OS-level keychain prompts just to discover no keys."""
    monkeypatch.delenv("CODNA_KEYS_FILE", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(cli, "__file__", "/venv/lib/python3.12/site-packages/codna/cli.py")

    from codna import keystore

    def forbidden_keyring():
        raise AssertionError("clean install runtime key discovery must not touch OS keychain")

    monkeypatch.setattr(keystore, "_keyring", forbidden_keyring)

    assert cli._runtime_keys() == {}


def test_runtime_keys_reads_keys_txt_in_source_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("CODNA_KEYS_FILE", raising=False)
    repo = tmp_path / "codna-repo"
    (repo / "cli" / "codna").mkdir(parents=True)
    (repo / "keys.txt").write_text("ANTHROPIC_API_KEY=sk-dev\n", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(repo / "cli" / "codna" / "cli.py"))
    from codna import keystore
    monkeypatch.setattr(keystore, "config_values", dict)
    keys = cli._runtime_keys()
    assert "ANTHROPIC_API_KEY" in keys and keys["ANTHROPIC_API_KEY"].source == "keys_txt"


def test_runtime_keys_can_skip_keychain_for_passive_commands(monkeypatch, tmp_path):
    monkeypatch.delenv("CODNA_KEYS_FILE", raising=False)
    repo = tmp_path / "codna-repo"
    (repo / "cli" / "codna").mkdir(parents=True)
    (repo / "keys.txt").write_text("ANTHROPIC_API_KEY=sk-dev\n", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(repo / "cli" / "codna" / "cli.py"))

    from codna import keystore

    def forbidden_keychain_read():
        raise AssertionError("passive commands must not read OS keychain secrets")

    monkeypatch.setattr(keystore, "config_values", forbidden_keychain_read)

    keys = cli._runtime_keys(include_keychain=False)

    assert "ANTHROPIC_API_KEY" in keys
