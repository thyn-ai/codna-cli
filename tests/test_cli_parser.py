"""CLI parser wiring (A–Z test plan: surfaces are thin adapters over the secure pipeline)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tomllib

import codna.cli as cli_module
from codna.cli import (
    build_parser,
    cmd_doctor,
    cmd_fix,
    cmd_mcp,
    cmd_secure,
    cmd_secure_open_pr,
    cmd_triage,
)


def test_secure_subcommand_wired():
    args = build_parser().parse_args(["secure", "myrepo", "--from-sarif", "results.sarif"])
    assert args.func is cmd_secure
    assert args.repo == "myrepo"
    assert args.from_sarif == "results.sarif"
    assert args.engine == "local"
    assert args.open_pr is False and args.fix is False


def test_secure_open_pr_and_verification_flags():
    args = build_parser().parse_args(
        ["secure", "--from-sarif", "r.sarif", "--verification", "codna-security.yaml", "--open-pr"]
    )
    assert args.from_sarif == "r.sarif"
    assert args.verification == "codna-security.yaml"
    assert args.open_pr is True


def test_secure_evidence_dir_flags():
    parser = build_parser()
    args = parser.parse_args([
        "secure", ".", "--from-sarif", "results.sarif", "--engine", "remote",
        "--fix", "--verification", "codna-security.yaml", "--evidence-dir", ".codna/evidence",
    ])
    assert args.fix is True
    assert args.evidence_dir == ".codna/evidence"


def test_secure_fix_exit_code_fails_when_eligible_findings_remain_unremediated():
    assert cli_module._secure_fix_exit_code(eligible=0, remediated=0) == 0
    assert cli_module._secure_fix_exit_code(eligible=1, remediated=1) == 0
    assert cli_module._secure_fix_exit_code(eligible=2, remediated=1) == 1
    assert cli_module._secure_fix_exit_code(eligible=1, remediated=0) == 1


def test_secure_open_pr_subcommand_wired():
    args = build_parser().parse_args(
        ["secure-open-pr", "--evidence", "evidence/", "--repo-slug", "o/r", "--base-branch", "main"]
    )
    assert args.func is cmd_secure_open_pr
    assert args.evidence == "evidence/"
    assert args.repo_slug == "o/r"


def test_secure_repo_slug_flag():
    args = build_parser().parse_args(
        ["secure", ".", "--from-sarif", "r.sarif", "--open-pr", "--repo-slug", "o/r"]
    )
    assert args.repo_slug == "o/r" and args.open_pr is True


def test_existing_subcommands_still_parse():
    p = build_parser()
    assert p.parse_args(["triage", "."]).func is cmd_triage
    assert p.parse_args(["fix", ".", "--issue", "x"]).func is cmd_fix
    assert p.parse_args(["doctor"]).func is cmd_doctor
    assert p.parse_args(["mcp"]).func is cmd_mcp


def test_main_formats_coded_errors_without_traceback(monkeypatch, capsys):
    class CodedError(RuntimeError):
        code = "local_repository_import_failed"
        details = {"reason": "missing apps package"}

    def fail(_args):
        raise CodedError("Codna could not import the local SDK.")

    class FakeParser:
        def parse_args(self, _argv):
            return SimpleNamespace(func=fail)

    monkeypatch.setattr(cli_module, "build_parser", lambda: FakeParser())

    rc = cli_module.main(["triage", "."])

    assert rc == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "error": {
            "code": "local_repository_import_failed",
            "message": "Codna could not import the local SDK.",
            "details": {"reason": "missing apps package"},
        }
    }


def test_main_formats_runtime_config_errors_without_die(monkeypatch, capsys):
    from codna import keystore

    def forbidden_keychain_read():
        raise AssertionError("runtime config errors for passive commands must not read OS keychain secrets")

    monkeypatch.setenv("CODNA_ENGINE_URL", "http://127.0.0.1:9999")
    monkeypatch.delenv("CODNA_ALLOW_LOOPBACK_ENGINE_URL", raising=False)
    monkeypatch.setattr(keystore, "config_values", forbidden_keychain_read)

    rc = cli_module.main(["triage", "."])

    assert rc == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "loopback_override_rejected"
    assert payload["error"]["details"]["url"] == "http://127.0.0.1:9999"


def test_mcp_is_an_optional_extra_not_a_base_dependency():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]

    # `mcp` pulls a web stack (starlette/uvicorn); keep it OUT of a plain `pip install codna` so the
    # base install stays lean and never clashes with a user's existing starlette/fastapi.
    assert not any(d == "mcp" or d.startswith("mcp>") or d.startswith("mcp<") or d.startswith("mcp=") for d in dependencies)
    # ...and available on demand via `pip install codna[mcp]`.
    assert "mcp>=1.2.0,<2" in extras["mcp"]


def test_github_action_uses_cli_without_forcing_remote_engine_or_hiding_failures():
    action = Path(__file__).resolve().parents[2] / "action.yml"
    text = action.read_text(encoding="utf-8")

    assert 'default: ""' in text
    # Since codna 0.1.35 the published wheel bundles a self-contained sidecar (Bun compiled in),
    # so the Action no longer sets up a JS runtime.
    assert "oven-sh/setup-bun" not in text
    assert "bun-version" not in text
    assert 'pipx install --pip-args=--only-binary=codna --force "$package_spec"' in text
    assert "bun run build:sdk" not in text
    assert "bash algenta/postbuild.sh" not in text
    assert "CODNA_SIDECAR_DIR" not in text
    assert 'CODNA_AGENT_CORE_READY_TIMEOUT_SECONDS: "120"' in text
    assert "export CODNA_ENGINE_URL" not in text
    assert "CODNA_ENGINE_URL: ${{ inputs.engine-url }}" not in text
    assert 'out="$(codna "${args[@]}" 2>&1)" || true' not in text
    assert 'exit "$status"' in text


def test_secure_open_pr_missing_evidence_is_user_error(capsys):
    rc = cli_module.main([
        "secure-open-pr",
        "--evidence",
        "/tmp/codna-missing-evidence-bundle",
        "--repo-slug",
        "owner/repo",
    ])

    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "could not read evidence bundle" in captured.err


def test_deterministic_commands_do_not_require_model_key_from_init_config(monkeypatch, tmp_path, capsys):
    from codna import scaffold

    scaffold.init_project(tmp_path, with_agents=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CODNA_MODEL_KEY", raising=False)

    called = {}

    def fake_triage(_args):
        called["triage"] = True
        return 0

    monkeypatch.setattr(cli_module, "cmd_triage", fake_triage)

    rc = cli_module.main(["triage", "."])

    assert rc == 0
    assert called == {"triage": True}
    assert "CODNA_REQUIRE_EGRESS_DENY" not in os.environ
    assert "config_error" not in capsys.readouterr().err


def test_fix_does_not_require_model_key_from_default_init_config(monkeypatch, tmp_path, capsys):
    from codna import scaffold

    scaffold.init_project(tmp_path, with_agents=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CODNA_MODEL_KEY", raising=False)

    called = {}

    def fake_fix(_args):
        called["fix"] = True
        return 0

    monkeypatch.setattr(cli_module, "cmd_fix", fake_fix)

    rc = cli_module.main(["fix", ".", "--issue", "broken"])

    assert rc == 0
    assert called == {"fix": True}
    assert "config_error" not in capsys.readouterr().err


def test_fix_requires_model_key_from_explicit_model_config(monkeypatch, tmp_path, capsys):
    (tmp_path / "codna.yaml").write_text(
        "model:\n  provider: openai\n  key: env:CODNA_MODEL_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CODNA_MODEL_KEY", raising=False)

    rc = cli_module.main(["fix", ".", "--issue", "broken"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "config_error" in err
    assert "CODNA_MODEL_KEY" in err


def test_secure_missing_sarif_is_user_error(tmp_path, capsys):
    rc = cli_module.main([
        "secure",
        str(tmp_path),
        "--from-sarif",
        str(tmp_path / "missing.sarif"),
        "--engine",
        "local",
        "--json",
    ])

    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "could not read SARIF" in captured.err


def test_secure_malformed_sarif_is_user_error(tmp_path, capsys):
    sarif = tmp_path / "bad.sarif"
    sarif.write_text("{not json", encoding="utf-8")

    rc = cli_module.main([
        "secure",
        str(tmp_path),
        "--from-sarif",
        str(sarif),
        "--engine",
        "local",
        "--json",
    ])

    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "could not ingest SARIF" in captured.err
    assert "malformed SARIF JSON" in captured.err


def test_secure_invalid_manifest_is_user_error(tmp_path, capsys):
    sarif = tmp_path / "ok.sarif"
    sarif.write_text(
        json.dumps({
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {"driver": {"name": "CodeQL", "version": "2.15.0", "rules": []}},
                "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
                "results": [],
            }],
        }),
        encoding="utf-8",
    )
    manifest = tmp_path / "codna-security.yaml"
    manifest.write_text(
        "scanner:\n  id: smoke\n  command: [true]\n  output: ok.sarif\n",
        encoding="utf-8",
    )

    rc = cli_module.main([
        "secure",
        str(tmp_path),
        "--from-sarif",
        str(sarif),
        "--engine",
        "local",
        "--fix",
        "--verification",
        str(manifest),
    ])

    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "invalid verification manifest" in captured.err
    assert "scanner.command[0]" in captured.err
