"""The bridge the IDE calls: one JSON in, one JSON out, and no silent answers.

The bridge exists so the IDE does not need a second rule engine in TypeScript. These tests hold it to
the two properties that makes it safe to call from another process: it always ANSWERS (never a
traceback on stdout), and a refusal is a structured object with a code rather than an empty string
that a host could mistake for "this repository has no rules".
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from codna import rules_bridge

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rules_repo"


def write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def mdc(description="", globs="", always=None, body="BODY"):
    lines = ["---"]
    if description:
        lines.append(f"description: {description}")
    if globs:
        lines.append(f"globs: {globs}")
    if always is not None:
        lines.append(f"alwaysApply: {'true' if always else 'false'}")
    lines.append("---")
    return "\n".join(lines) + "\n" + body + "\n"


# ---------------------------------------------------------------------------------------------
# handle(): the pure half
# ---------------------------------------------------------------------------------------------

def test_handle_returns_the_engine_answer():
    response = rules_bridge.handle({"repo": str(FIXTURE), "files_in_context": ["src/a.tsx"]})
    assert response["ok"] is True
    assert "Python: use ruff" in response["text"]
    assert "TypeScript: prefer" in response["text"]
    assert "CSS: use custom properties" not in response["text"]
    assert response["used_bytes"] <= response["budget_bytes"]


def test_handle_reports_the_kept_and_dropped_names():
    response = rules_bridge.handle({"repo": str(FIXTURE), "files_in_context": ["src/a.tsx"]})
    kept = {rule["name"] for rule in response["kept"]}
    assert "python" in kept and "typescript" in kept
    assert [rule["name"] for rule in response["catalogued"]] == ["migrations"]
    assert [d["source_path"] for d in response["dropped"]] == [".cursor/rules/python.mdc"]


def test_handle_distinguishes_unknown_files_from_an_empty_set():
    """The two answers differ, and the bridge must carry that difference through."""
    unknown = rules_bridge.handle({"repo": str(FIXTURE)})
    empty = rules_bridge.handle({"repo": str(FIXTURE), "files_in_context": []})
    assert unknown["files_known"] is False
    assert empty["files_known"] is True
    # An unknown set keeps the glob rules; a known-empty one excludes them.
    assert any(rule["mode"] == "glob" for rule in unknown["kept"])
    assert not any(rule["mode"] == "glob" for rule in empty["kept"])


def test_handle_honours_max_bytes_and_reports_drops(tmp_path):
    for index in range(10):
        write(tmp_path, f".codna/rules/r{index}.mdc", mdc(always=True, body="y" * 400))
    response = rules_bridge.handle({"repo": str(tmp_path), "max_bytes": 600})
    assert response["ok"] is True
    assert response["used_bytes"] <= 600
    assert response["dropped"], "a tight budget must be reported, not silently applied"


def test_handle_pulls_a_catalogued_rule_via_explicit_mentions():
    response = rules_bridge.handle({
        "repo": str(FIXTURE), "explicit_mentions": ["@migrations"],
    })
    assert "make migrate" in response["text"]


@pytest.mark.parametrize(
    ("request_obj", "code"),
    [
        ([], "bridge_bad_request"),
        ({}, "bridge_bad_request"),
        ({"repo": ""}, "bridge_bad_request"),
        ({"repo": "   "}, "bridge_bad_request"),
        ({"repo": "/x", "files_in_context": "src/a.ts"}, "bridge_bad_request"),
        ({"repo": "/x", "explicit_mentions": "@python"}, "bridge_bad_request"),
        ({"repo": "/x", "max_bytes": -1}, "bridge_bad_request"),
        ({"repo": "/x", "max_bytes": "lots"}, "bridge_bad_request"),
    ],
)
def test_handle_refuses_malformed_requests_structurally(request_obj, code):
    response = rules_bridge.handle(request_obj)
    assert response["ok"] is False
    assert response["error"]["code"] == code


def test_handle_reports_a_missing_repository_rather_than_returning_nothing(tmp_path):
    response = rules_bridge.handle({"repo": str(tmp_path / "nope")})
    # Still `ok` — a directory that does not exist is a discovery warning, not a bridge failure — but
    # the warning is present, so a host can tell "no rules" from "wrong path".
    assert response["ok"] is True
    assert response["text"] == ""
    assert any(w["code"] == "rules_root_missing" for w in response["warnings"])


# ---------------------------------------------------------------------------------------------
# main(): the process contract
# ---------------------------------------------------------------------------------------------

def _run(request_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "codna.rules_bridge"],
        input=request_text,
        text=True,
        capture_output=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PATH": "/usr/bin:/bin"},
        timeout=60,
    )


def test_main_writes_one_json_object_and_exits_zero():
    completed = _run(json.dumps({"repo": str(FIXTURE), "files_in_context": ["src/a.tsx"]}))
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["kept"]


def test_main_exits_one_on_a_refusal_and_still_writes_json():
    completed = _run(json.dumps({"repo": ""}))
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "bridge_bad_request"


def test_main_reports_invalid_json_rather_than_tracebacking():
    completed = _run("{not json")
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["error"]["code"] == "bridge_bad_json"
    assert "Traceback" not in completed.stdout


def test_main_reports_a_missing_repository_as_a_warning_not_a_crash():
    completed = _run(json.dumps({"repo": "/definitely/not/a/repo"}))
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["warnings"]
