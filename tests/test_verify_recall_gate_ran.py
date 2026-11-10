from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import typing

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_recall_gate_ran.py"

REQUIRED = (
    "test_recall_happy_path",
    "test_declared_site_dir_comes_from_the_manifest",
    "test_declared_site_dir_is_activated_even_behind_a_shadowing_telys",
)


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_recall_gate_ran", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path, cases: list[str]) -> Path:
    """A JUnit report shaped like pytest's: one `testcase` per entry, `@child` selects an outcome."""
    body = []
    for entry in cases:
        if "@" in entry:
            name, outcome = entry.split("@", 1)
            body.append(f'<testcase classname="tests.test_mcp_recall" name="{name}"><{outcome}/></testcase>')
        else:
            body.append(f'<testcase classname="tests.test_mcp_recall" name="{entry}"/>')
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        + "".join(body)
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return path


# ── happy path ────────────────────────────────────────────────────────────────────────────────────
def test_gate_passes_when_every_required_test_ran_and_passed(tmp_path: Path):
    tool = _load_script()
    report = _write_report(tmp_path / "recall.xml", [*REQUIRED, "test_recall_requires_query"])

    result = tool.verify(report)

    assert result["ok"] is True
    assert result["executed"] == list(REQUIRED)
    assert result["test_cases"] == 4


def test_gate_ignores_the_deselected_style_node_id_reporting(tmp_path: Path):
    """THE REGRESSION. `-q` without `pytest-relaxed` prints dots, not test names.

    The old shell guard grepped the TEXT output for humanised names, so it failed on a green suite on
    any machine without that plugin. The report below carries exactly what such a run produces and
    must pass: the assertion is about outcomes, not about how a reporter formatted them.
    """
    tool = _load_script()
    report = _write_report(tmp_path / "recall.xml", [*REQUIRED])

    result = tool.verify(report)

    assert result["ok"] is True


# ── it must fail closed ──────────────────────────────────────────────────────────────────────────
def test_gate_fails_when_the_report_is_absent(tmp_path: Path):
    tool = _load_script()

    with pytest.raises(tool.RecallGateError) as excinfo:
        tool.verify(tmp_path / "missing.xml")

    assert excinfo.value.code == "recall_report_missing"


def test_gate_fails_when_required_test_is_missing_from_the_report(tmp_path: Path):
    tool = _load_script()
    report = _write_report(tmp_path / "recall.xml", ["test_recall_requires_query"])

    with pytest.raises(tool.RecallGateError) as excinfo:
        tool.verify(report)

    assert excinfo.value.code == "recall_required_tests_missing"
    assert "test_recall_happy_path" in str(excinfo.value)


def test_gate_fails_when_a_required_test_skipped(tmp_path: Path):
    """A skip is precisely the vacuous green this job exists to prevent."""
    tool = _load_script()
    report = _write_report(
        tmp_path / "recall.xml",
        [f"{REQUIRED[0]}@skipped", *REQUIRED[1:]],
    )

    with pytest.raises(tool.RecallGateError) as excinfo:
        tool.verify(report)

    assert excinfo.value.code == "recall_required_tests_did_not_pass"
    assert "skipped" in str(excinfo.value)


def test_gate_fails_when_a_required_test_failed(tmp_path: Path):
    tool = _load_script()
    report = _write_report(
        tmp_path / "recall.xml",
        [f"{REQUIRED[0]}@failure", *REQUIRED[1:]],
    )

    with pytest.raises(tool.RecallGateError) as excinfo:
        tool.verify(report)

    assert excinfo.value.code == "recall_required_tests_did_not_pass"


def test_gate_fails_when_an_unrelated_recall_test_failed(tmp_path: Path):
    tool = _load_script()
    report = _write_report(
        tmp_path / "recall.xml",
        [*REQUIRED, "test_recall_requires_query@failure"],
    )

    with pytest.raises(tool.RecallGateError) as excinfo:
        tool.verify(report)

    assert excinfo.value.code == "recall_suite_failed"


def test_gate_fails_on_a_malformed_report(tmp_path: Path):
    tool = _load_script()
    report = tmp_path / "recall.xml"
    report.write_text("<testsuites><testsuite>", encoding="utf-8")

    with pytest.raises(tool.RecallGateError) as excinfo:
        tool.verify(report)

    assert excinfo.value.code == "recall_report_malformed"


def test_gate_refuses_a_report_declaring_xml_entities(tmp_path: Path):
    """XXE / billion-laughs is refused, not expanded — the report is parsed with `defusedxml`."""
    tool = _load_script()
    report = tmp_path / "recall.xml"
    report.write_text(
        '<?xml version="1.0"?>'
        '<!DOCTYPE x [<!ENTITY boom "expanded">]>'
        '<testsuites><testsuite><testcase name="&boom;"/></testsuite></testsuites>',
        encoding="utf-8",
    )

    with pytest.raises(tool.RecallGateError) as excinfo:
        tool.verify(report)

    assert excinfo.value.code == "recall_report_unsafe"


def test_type_annotations_are_resolvable_at_runtime():
    """Annotations must name a type that actually exists, not just one PEP 563 defers.

    `defusedxml.ElementTree` does not re-export `Element` (its `__all__` is the parsing API plus
    `ParseError`), so annotating with `ElementTree.Element` looks fine only because
    `from __future__ import annotations` defers evaluation — `typing.get_type_hints()` then raises
    `AttributeError`. Parse through `defusedxml`, but annotate with the stdlib type it returns.
    """
    tool = _load_script()

    for func in (tool._parse, tool._test_cases, tool._outcomes):  # noqa: SLF001 — the annotations ARE the subject
        assert typing.get_type_hints(func) is not None


# ── CLI contract ────────────────────────────────────────────────────────────────────────────────
def test_cli_exits_zero_and_prints_json_on_success(tmp_path: Path, capsys):
    tool = _load_script()
    report = _write_report(tmp_path / "recall.xml", [*REQUIRED])

    assert tool.main(["--junit-report", str(report)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_cli_exits_nonzero_and_names_the_code_on_failure(tmp_path: Path, capsys):
    tool = _load_script()
    report = _write_report(tmp_path / "recall.xml", ["test_recall_requires_query"])

    assert tool.main(["--junit-report", str(report)]) == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "recall_required_tests_missing"
    assert payload["annotation"].startswith("::error title=Recall gate did not run::")
