"""Sandbox tests (A–Z test plan: PRIV-01/02/03 privilege enforcement, timeout, cwd)."""
from __future__ import annotations

import os

import pytest

from codna.sandbox import (
    PrivilegeSeparationError,
    Sandbox,
    assert_no_write_credentials,
    scrub_env,
)

PATH_ONLY = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def test_scrub_env_removes_write_tokens_by_name_and_value():
    env = {"PATH": "/bin", "GITHUB_TOKEN": "x", "CODNA_GITHUB_TOKEN": "y",
           "SOME_VAR": "ghp_" + "a" * 36, "SAFE": "value"}
    scrubbed = scrub_env(env)
    assert "GITHUB_TOKEN" not in scrubbed
    assert "CODNA_GITHUB_TOKEN" not in scrubbed
    assert "SOME_VAR" not in scrubbed  # removed by value pattern (looks like a PAT)
    assert scrubbed["SAFE"] == "value" and scrubbed["PATH"] == "/bin"


def test_assert_no_write_credentials():
    assert_no_write_credentials({"PATH": "/bin"})  # no raise
    with pytest.raises(PrivilegeSeparationError):
        assert_no_write_credentials({"GH_TOKEN": "abc"})
    with pytest.raises(PrivilegeSeparationError):
        assert_no_write_credentials({"WHATEVER": "github_pat_" + "a" * 30})


def test_sandbox_construction_rejects_write_token():
    with pytest.raises(PrivilegeSeparationError):
        Sandbox(env={"GITHUB_TOKEN": "ghp_" + "z" * 36})


def test_sandbox_runs_command_in_cwd(tmp_path):
    sb = Sandbox(env=dict(PATH_ONLY))
    res = sb.run(["python3", "-c", "import os;print(os.getcwd())"], cwd=str(tmp_path))
    assert res.returncode == 0
    assert res.cwd == str(tmp_path)
    assert res.network == "deny" and sb.denies_network is True
    assert os.path.realpath(res.stdout.strip()) == os.path.realpath(str(tmp_path))


def test_sandbox_accepted_exit_codes(tmp_path):
    sb = Sandbox(env=dict(PATH_ONLY))
    res = sb.run(["python3", "-c", "import sys;sys.exit(3)"], cwd=str(tmp_path))
    assert res.returncode == 3
    assert res.accepted((0,)) is False
    assert res.accepted((0, 3)) is True


def test_sandbox_timeout(tmp_path):
    sb = Sandbox(env=dict(PATH_ONLY), timeout_seconds=1)
    res = sb.run(["python3", "-c", "import time;time.sleep(5)"], cwd=str(tmp_path))
    assert res.timed_out is True
    assert res.accepted((0,)) is False
