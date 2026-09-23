from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import benchmark_suite as bench  # noqa: E402
import benchmark_cases  # noqa: E402


def test_default_engines_are_current_enterprise_comparison() -> None:
    assert bench._parse_engines(bench.DEFAULT_ENGINES) == [
        "codna-nomem",
        "codna",
        "cline",
        "cursor",
    ]


def test_default_engines_are_supported_by_security_mode() -> None:
    security_engines = [engine for engine in bench._parse_engines(bench.DEFAULT_ENGINES) if engine != "codna-nomem"]

    assert set(security_engines).issubset(bench.SEC_RUNNERS)


def test_parse_engines_rejects_unknown_instead_of_silently_dropping() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        bench._parse_engines("codna,not-real,cursor")


def test_random_selection_rejects_over_claimed_case_count() -> None:
    with pytest.raises(ValueError, match="only .* curated benchmark cases"):
        bench._select_random_jobs(len(bench.CURATED) + 1, seed=1)


def test_case_manifest_supports_requested_100_case_run() -> None:
    assert len(bench.CURATED) >= 100
    assert len({repo_url for repo_url, _issue in bench.CURATED}) == len(bench.CURATED)


def test_random_selection_is_seeded_and_reproducible() -> None:
    first = bench._select_random_jobs(3, seed=7)
    second = bench._select_random_jobs(3, seed=7)

    assert first == second
    assert len(first) == 3


def test_soak_case_parser_extracts_repo_and_issue(tmp_path: Path) -> None:
    case_file = tmp_path / "case.md"
    case_file.write_text(
        "\n".join(
            [
                "# report",
                "",
                "_repo: https://github.com/example/project " + "\u00b7" + " issue: fix the broken edge case_",
            ]
        ),
        encoding="utf-8",
    )

    assert benchmark_cases.load_tracked_soak_cases(tmp_path) == [
        ("https://github.com/example/project", "fix the broken edge case")
    ]


def test_case_manifest_dedupes_by_repo_url(tmp_path: Path) -> None:
    case_file = tmp_path / "duplicate.md"
    case_file.write_text(
        "_repo: https://github.com/psf/requests " + "\u00b7" + " issue: duplicate requests issue_",
        encoding="utf-8",
    )

    cases = benchmark_cases.benchmark_cases(include_tracked_soak=False)
    cases.extend(benchmark_cases.load_tracked_soak_cases(tmp_path))
    deduped = benchmark_cases._dedupe_by_repo_url(cases)  # noqa: SLF001

    assert [repo_url for repo_url, _issue in deduped].count("https://github.com/psf/requests") == 1


def test_run_returns_127_for_missing_executable(tmp_path: Path) -> None:
    rc, out, err, _dt = bench._run(["definitely-not-a-real-codna-bench-command"], cwd=str(tmp_path))

    assert rc == 127
    assert out == ""
    assert "No such file" in err


def test_ensure_codna_runtime_uses_published_doctor_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(cmd, *, timeout):
        calls.append((cmd, timeout))
        return 0, "Codna runtime: healthy", "", 0.2

    monkeypatch.delenv("CODNA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_BASE_URL", raising=False)
    monkeypatch.setattr(bench, "CODNA", "codna")
    monkeypatch.setattr(bench, "_run", fake_run)

    assert bench._ensure_codna_runtime() is None
    assert calls == [(["codna", "doctor", "--start-stack"], 120)]


def test_codna_fix_reports_planner_token_split(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "runtime_model": "claude-sonnet-4-6",
        "impacted_symbols": ["src/payment.py::charge"],
        "context": {
            "raw_token_estimate": 10000,
            "evidence_bundle_tokens": 400,
            "reduction_ratio": 25,
        },
        "planner_usage": {
            "input_tokens": 1234,
            "output_tokens": 56,
            "cache_read_tokens": 78,
        },
        "cost_usd": 0.01,
        "patch_ref": "patch_1",
        "root_cause": "missing retry",
    }

    def fake_run(*_args, **_kwargs):
        return 0, json.dumps(payload), "", 1.2

    monkeypatch.setenv("CODNA_API_KEY", "codna_test")
    monkeypatch.setattr(bench, "_ensure_codna_runtime", lambda: None)
    monkeypatch.setattr(bench, "_run", fake_run)

    row = bench._codna_fix("/tmp/repo", "bug", use_memory=False)

    assert row["engine"] == "Codna " + "\u2212" + "Telys"
    assert row["context_in"] == 1234
    assert row["out_tokens"] == 56
    assert row["cache_read"] == 78
    assert row["total_tokens"] == 1290
    assert row["evidence_tokens"] == 400
    assert row["context_note"] == "10,000" + "\u2192" + "400 (25" + "\u00d7" + " smaller)"


def test_parse_verify_cmd_preserves_shell_style_quoting() -> None:
    assert bench._parse_verify_cmd("pytest -q 'tests/unit path'") == ["pytest", "-q", "tests/unit path"]


def test_parse_verify_cmd_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="command cannot be empty"):
        bench._parse_verify_cmd(" ")


def test_run_precheck_accepts_failing_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench, "_run", lambda *_args, **_kwargs: (1, "", "expected failure", 0.3))

    result = bench._run_precheck(str(tmp_path), ["pytest", "-q"])

    assert result["precheck_ok"] is True
    assert result["precheck"] == "fail-first"
    assert result["precheck_notes"] == "expected failure"


def test_run_precheck_rejects_passing_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench, "_run", lambda *_args, **_kwargs: (0, "passed", "", 0.2))

    result = bench._run_precheck(str(tmp_path), ["pytest", "-q"])

    assert result["precheck_ok"] is False
    assert result["precheck"] == "passed"
    assert "not reproduced" in result["precheck_notes"]


def test_bench_one_skips_engines_when_precheck_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("fixture", encoding="utf-8")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(bench, "_run", lambda *_args, **_kwargs: (0, "passed", "", 0.1))

    name, rows = bench._bench_one(str(repo), "bug", ["cursor"], None, str(workdir), ["pytest", "-q"], None)

    assert name == "repo"
    assert rows == [
        {
            "engine": "cursor",
            "status": "skipped",
            "notes": "precheck passed; issue was not reproduced",
            "precheck": "passed",
            "precheck_notes": "precheck passed; issue was not reproduced",
            "precheck_time_s": 0.1,
            "accuracy_available": False,
            "verified": None,
            "verification": "n/a",
            "verification_notes": "precheck did not establish a failing baseline",
            "fix_result": "n/a",
        }
    ]


def test_attach_verification_runs_for_local_patch_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(cmd, cwd=None, timeout=1800, env=None, stdin=None):
        calls.append((cmd, cwd, timeout, env, stdin))
        return 0, "passed", "", 0.2

    monkeypatch.setattr(bench, "_run", fake_run)
    row = bench._attach_verification({"status": "ok"}, "cursor", str(tmp_path), ["pytest", "-q"])

    assert row["accuracy_available"] is True
    assert row["verified"] is True
    assert row["verification"] == "pass"
    assert row["fix_result"] == "unqualified"
    assert calls == [(["pytest", "-q"], str(tmp_path), 300, None, None)]


def test_attach_verification_marks_fail_first_pass_as_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench, "_run", lambda *_args, **_kwargs: (0, "passed", "", 0.2))

    row = bench._attach_verification({"status": "ok", "precheck": "fail-first"}, "cursor", str(tmp_path), ["pytest"])

    assert row["verification"] == "pass"
    assert row["fix_result"] == "verified"


def test_attach_verification_keeps_codna_inspect_mode_unscored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CODNA_BENCH_MODE", raising=False)

    row = bench._attach_verification({"status": "ok"}, "codna", str(tmp_path), ["pytest", "-q"])

    assert row["accuracy_available"] is False
    assert row["verified"] is None
    assert row["verification"] == "n/a"
    assert "patch_ref is not applied" in row["verification_notes"]


def test_render_includes_split_token_and_accuracy_columns() -> None:
    report = bench.render(
        "repo",
        "https://github.com/example/repo",
        "bug",
        [
            {
                "engine": "Codna +Telys",
                "model": "claude-sonnet-4-6",
                "localized": "src/payment.py",
                "context_in": 1234,
                "out_tokens": 56,
                "cache_read": 78,
                "total_tokens": 1290,
                "evidence_tokens": 400,
                "time_s": 12,
                "cost_usd": 0.01,
                "status": "ok",
                "memory": "10 sym idx (0.1s) - 2 recalled",
            }
        ],
    )

    assert "| Memory (Telys) | Time | Cost | Precheck | Final verify | Fix verified | Status |" in report
    assert (
        "| Codna +Telys | claude-sonnet-4-6 | src/payment.py | 1,234 | 56 | 78 | 1,290 | 400 |"
    ) in report
    assert "| 12s | $0.010 | n/a | n/a | n/a | ok |" in report
