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
# SKILLS IN THE SAME ANSWER — the wiring assertion
# ---------------------------------------------------------------------------------------------
#
# `test_skill_catalogue_reaches_session` lives HERE, at the bridge, because the bridge's response IS
# the create-session body's `injected_context`. #713 nearly shipped a chat with no project rules
# because the text was produced and then never put on the wire; the assertion that would have caught
# it is "the text the producer returned is in the body the session is created with". The same
# assertion, one layer down, is exactly what this test makes: `handle()`'s `text` is what a host puts
# on `injected_context`, so a catalogue that is not in `text` never reaches a session.

SKILLS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skills_repo"


def skill_md(description: str = "", paths: str = "", manual: bool = False, body: str = "BODY") -> str:
    lines = ["---"]
    if description:
        lines.append(f"description: {description}")
    if paths:
        lines.append(f"paths: {paths}")
    if manual:
        lines.append("disable-model-invocation: true")
    lines.append("---")
    return "\n".join(lines) + "\n" + body + "\n"


def test_skill_catalogue_reaches_session(tmp_path):
    """The catalogue is in `text` — i.e. in what goes on `injected_context` — not merely discovered.

    Three claims, and the third is the one that matters:
      1. a `.cursor/skills` skill is discovered at all;
      2. its catalogue LINE is in the response text;
      3. its BODY is not — the whole point of progressive disclosure.
    """
    (tmp_path / ".cursor" / "skills" / "pdf").mkdir(parents=True)
    (tmp_path / ".cursor" / "skills" / "pdf" / "SKILL.md").write_text(
        skill_md(description="fill a PDF form", paths="src/**/*.ts", body="SECRET-BODY-TEXT"),
        encoding="utf-8",
    )

    # `home` is HERMETIC on purpose: the user layer must not make this assertion depend on what the
    # machine running the suite happens to have in `~/.cursor/skills`. The product's real call omits it.
    response = rules_bridge.handle({
        "repo": str(tmp_path), "files_in_context": ["src/app.ts"], "home": str(tmp_path / "no-home"),
    })

    assert response["ok"] is True
    # 1. discovered, and named in the structured half a host can show.
    assert [skill["name"] for skill in response["skills"]] == ["pdf"]
    assert response["skills"][0]["scope"] == "cursor"
    # 2. the CATALOGUE LINE is in the text that becomes `injected_context`.
    assert "- pdf: fill a PDF form — paths: src/**/*.ts" in response["text"]
    assert "codna_load_skill" in response["text"]
    # 3. the BODY is not.
    assert "SECRET-BODY-TEXT" not in response["text"]


def test_a_scoped_skill_is_absent_from_the_text_for_a_docs_only_context():
    """The `paths` decision reaches the wire, not just the selector."""
    home = str(SKILLS_FIXTURE / "no-home")
    code = rules_bridge.handle(
        {"repo": str(SKILLS_FIXTURE), "files_in_context": ["src/app.ts"], "home": home})
    docs = rules_bridge.handle(
        {"repo": str(SKILLS_FIXTURE), "files_in_context": ["docs/a.md"], "home": home})

    assert "- pdf:" in code["text"]
    assert "- pdf:" not in docs["text"]
    # The unscoped ones stay, so scoping narrows and does not switch the repository off.
    assert "- api:" in docs["text"]


def test_rules_and_skills_share_one_budget_in_the_bridge(tmp_path):
    (tmp_path / ".codna" / "rules").mkdir(parents=True)
    (tmp_path / ".codna" / "rules" / "python.mdc").write_text(
        "---\nalwaysApply: true\n---\n" + "R" * 300 + "\n", encoding="utf-8",
    )
    (tmp_path / ".cursor" / "skills" / "pdf").mkdir(parents=True)
    (tmp_path / ".cursor" / "skills" / "pdf" / "SKILL.md").write_text(
        skill_md(description="d"), encoding="utf-8",
    )

    response = rules_bridge.handle(
        {"repo": str(tmp_path), "max_bytes": 800, "home": str(tmp_path / "no-home")})

    assert response["used_bytes"] <= response["budget_bytes"] == 800
    assert response["skills_budget_bytes"] <= 800 - response["used_bytes"] + response["skills_used_bytes"]
    assert "- pdf:" in response["text"]


def test_include_skills_false_is_the_pre_skill_answer():
    """A host that has not adopted skills sends the request it always did and gets the old answer."""
    home = str(SKILLS_FIXTURE / "no-home")
    with_skills = rules_bridge.handle({"repo": str(SKILLS_FIXTURE), "home": home})
    without = rules_bridge.handle(
        {"repo": str(SKILLS_FIXTURE), "home": home, "include_skills": False})
    assert with_skills["skills"]
    assert without["skills"] == []
    assert "SKILLS AVAILABLE" not in without["text"]


@pytest.mark.parametrize("bad", ["yes", 1])
def test_include_skills_rejects_a_non_boolean(bad):
    response = rules_bridge.handle({"repo": "/x", "include_skills": bad})
    assert response["ok"] is False
    assert response["error"]["code"] == "bridge_bad_request"


def test_a_manual_only_skill_is_catalogued_and_marked_in_the_text():
    """`disable-model-invocation` reaches the wire as a marking on the line, not as an omission."""
    response = rules_bridge.handle(
        {"repo": str(SKILLS_FIXTURE), "home": str(SKILLS_FIXTURE / "no-home")})
    release = {skill["name"]: skill for skill in response["skills"]}["release"]
    assert release["mode"] == "manual"
    assert "[manual-only: /release]" in response["text"]


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
