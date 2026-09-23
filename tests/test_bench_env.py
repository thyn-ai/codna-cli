"""The bench scripts pass operator-supplied binaries / paths straight to ``subprocess``; ``bench_env``
validates them up front (allowlisted name on PATH, or an absolute existing executable) so an env
override can never smuggle an arbitrary command in, and a mis-set one fails with a clear message."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import bench_env  # noqa: E402


def _make_exe(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolve_bin_bare_allowlisted_name_resolves_through_path(tmp_path, monkeypatch):
    exe = _make_exe(tmp_path / "codna")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CODNA_BIN", raising=False)
    assert bench_env.resolve_bin("CODNA_BIN", "codna", allowed=("codna",)) == str(exe)


def test_resolve_bin_rejects_bare_name_outside_allowlist(tmp_path, monkeypatch):
    _make_exe(tmp_path / "rm")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("CODNA_BIN", "rm")
    with pytest.raises(SystemExit, match="not an allowed command name"):
        bench_env.resolve_bin("CODNA_BIN", "codna", allowed=("codna",))


def test_resolve_bin_rejects_bare_name_missing_from_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir: nothing resolvable
    monkeypatch.delenv("CODEX_BIN", raising=False)
    with pytest.raises(SystemExit, match="not found on PATH"):
        bench_env.resolve_bin("CODEX_BIN", "codex", allowed=("codex",))


def test_resolve_bin_accepts_absolute_existing_executable_and_expands_home(tmp_path, monkeypatch):
    exe = _make_exe(tmp_path / "cursor-agent")
    monkeypatch.setenv("CURSOR_BIN", str(exe))
    assert bench_env.resolve_bin("CURSOR_BIN", "cursor-agent", allowed=("cursor-agent",)) == str(exe)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CURSOR_BIN", raising=False)
    assert bench_env.resolve_bin("CURSOR_BIN", "~/cursor-agent", allowed=("cursor-agent",)) == str(exe)


def test_resolve_bin_rejects_relative_missing_and_non_executable_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("CODNA_BIN", os.path.join("bin", "codna"))
    with pytest.raises(SystemExit, match="must be absolute"):
        bench_env.resolve_bin("CODNA_BIN", "codna", allowed=("codna",))
    monkeypatch.setenv("CODNA_BIN", str(tmp_path / "missing" / "codna"))
    with pytest.raises(SystemExit, match="not an existing executable file"):
        bench_env.resolve_bin("CODNA_BIN", "codna", allowed=("codna",))
    plain = tmp_path / "codna"
    plain.write_text("not executable", encoding="utf-8")
    plain.chmod(0o644)
    monkeypatch.setenv("CODNA_BIN", str(plain))
    with pytest.raises(SystemExit, match="not an existing executable file"):
        bench_env.resolve_bin("CODNA_BIN", "codna", allowed=("codna",))


def test_model_id_accepts_identifiers_and_rejects_option_like_or_shell_like_values(monkeypatch):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    assert bench_env.model_id("CODEX_MODEL") == ""
    for ok in ("gpt-5.4", "openai/gpt-5", "claude-sonnet-4-6", "anthropic:claude_3"):
        monkeypatch.setenv("CODEX_MODEL", ok)
        assert bench_env.model_id("CODEX_MODEL") == ok
    for bad in ("--full-auto", "gpt 5", "gpt;rm -rf /", "$(id)", "a" * 129):
        monkeypatch.setenv("CODEX_MODEL", bad)
        with pytest.raises(SystemExit, match="not a model identifier"):
            bench_env.model_id("CODEX_MODEL")


def test_resolve_file_requires_an_existing_file(tmp_path, monkeypatch):
    manifest = tmp_path / "codna-security.yaml"
    manifest.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.delenv("SEC_MANIFEST", raising=False)
    assert bench_env.resolve_file("SEC_MANIFEST", str(manifest)) == str(manifest)
    monkeypatch.setenv("SEC_MANIFEST", str(tmp_path / "nope.yaml"))
    with pytest.raises(SystemExit, match="file not found"):
        bench_env.resolve_file("SEC_MANIFEST", str(manifest))


def test_bench_repos_root_uses_first_arg_and_requires_a_directory(tmp_path):
    assert bench_env.bench_repos_root([str(tmp_path)]) == str(tmp_path)
    assert bench_env.bench_repos_root([], default=str(tmp_path)) == str(tmp_path)
    with pytest.raises(SystemExit, match="is not a directory"):
        bench_env.bench_repos_root([str(tmp_path / "missing")])
