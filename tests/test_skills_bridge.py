"""``codna.skills_bridge`` — the catalogue as data, and the pull, over one stdin/stdout protocol.

The bridge exists so the IDE can offer ``/skill-name`` from the SAME reader the prompt path uses, and
these tests hold it to the two properties that make it safe to call from another process: it always
ANSWERS (a refusal is a structured object with a code, never an empty stdout a host could mistake for
"this repository has no skills"), and the catalogue it returns carries NO bodies — the body is the pull.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import unlistable_directory

from codna import skills_bridge

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skills_repo"
EMPTY_HOME = str(FIXTURE / "no-home")


def write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def skill_md(description="", paths="", manual=False, body="BODY"):
    lines = ["---"]
    if description:
        lines.append(f"description: {description}")
    if paths:
        lines.append(f"paths: {paths}")
    if manual:
        lines.append("disable-model-invocation: true")
    lines.append("---")
    return "\n".join(lines) + "\n" + body + "\n"


# ---------------------------------------------------------------------------------------------
# handle(): the pure half
# ---------------------------------------------------------------------------------------------

def test_the_catalogue_carries_structure_and_no_bodies():
    response = skills_bridge.handle(
        {"repo": str(FIXTURE), "files_in_context": ["src/app.ts"], "home": EMPTY_HOME},
    )
    assert response["ok"] is True
    names = [skill["name"] for skill in response["skills"]]
    assert names == ["release", "pdf", "api"]  # codna → cursor → agents
    # …and NOT one body. The catalogue is an index; the pull is how instructions arrive.
    assert "body" not in response["skills"][0]
    assert all("body" not in skill for skill in response["skills"])


def test_the_catalogue_carries_what_a_caller_needs_to_decide():
    response = skills_bridge.handle(
        {"repo": str(FIXTURE), "files_in_context": ["src/app.ts"], "home": EMPTY_HOME},
    )
    pdf = {skill["name"]: skill for skill in response["skills"]}["pdf"]
    assert pdf["description"] == "Fill and validate a PDF form from TypeScript"
    assert pdf["scope"] == "cursor"
    assert pdf["source_path"] == ".cursor/skills/pdf/SKILL.md"
    assert pdf["paths"] == ["src/**/*.ts"]
    assert pdf["mode"] == "auto"
    assert pdf["bundled"] == ["scripts/fill.ts"]


def test_paths_scoping_reaches_the_bridge():
    """The file set decides, through the same glob matcher a rule uses."""
    code = skills_bridge.handle(
        {"repo": str(FIXTURE), "files_in_context": ["src/app.ts"], "home": EMPTY_HOME},
    )
    docs = skills_bridge.handle(
        {"repo": str(FIXTURE), "files_in_context": ["docs/a.md"], "home": EMPTY_HOME},
    )
    assert "pdf" in [skill["name"] for skill in code["skills"]]
    assert "pdf" not in [skill["name"] for skill in docs["skills"]]


def test_an_unknown_file_set_keeps_a_scoped_skill():
    """`None` is "the caller did not say", which cannot rule a scoped skill out."""
    response = skills_bridge.handle({"repo": str(FIXTURE), "home": EMPTY_HOME})
    assert "pdf" in [skill["name"] for skill in response["skills"]]
    assert response["files_known"] is False


def test_the_budget_is_reported_not_applied_silently(tmp_path):
    for index in range(20):
        write(
            tmp_path,
            f".cursor/skills/s{index}/SKILL.md",
            skill_md(description=f"skill number {index} with a description of some length"),
        )
    response = skills_bridge.handle(
        {"repo": str(tmp_path), "home": str(tmp_path / "no-home"), "max_bytes": 400},
    )
    assert response["used_bytes"] <= response["budget_bytes"] == 400
    assert len(response["skills"]) + len(response["skills_dropped"]) == 20
    assert response["skills_dropped"]


def test_the_pull_returns_the_body_and_its_bundled_files():
    response = skills_bridge.handle(
        {"repo": str(FIXTURE), "skill": "pdf", "home": EMPTY_HOME},
    )
    assert response["ok"] is True
    assert response["name"] == "pdf"
    assert "Render the form" in response["body"]
    assert response["bundled"] == ["scripts/fill.ts"]


def test_the_pull_is_the_SAME_body_the_prompt_path_loads():
    """`/skill-name` and `codna_load_skill` must answer identically — one reader, two askers."""
    from codna.rules import load_skill

    response = skills_bridge.handle({"repo": str(FIXTURE), "skill": "/pdf", "home": EMPTY_HOME})
    direct = load_skill(str(FIXTURE), "pdf", home=EMPTY_HOME)
    assert direct is not None
    assert response["body"] == direct.body


def test_a_manual_only_skill_is_pullable_and_marked():
    response = skills_bridge.handle(
        {"repo": str(FIXTURE), "skill": "release", "home": EMPTY_HOME},
    )
    assert response["ok"] is True
    assert response["mode"] == "manual"
    assert response["body"]


def test_an_unknown_skill_is_a_structured_refusal():
    response = skills_bridge.handle({"repo": str(FIXTURE), "skill": "nope", "home": EMPTY_HOME})
    assert response["ok"] is False
    assert response["error"]["code"] == "skill_not_found"
    assert response["error"]["details"]["requested"] == "nope"


@pytest.mark.parametrize(
    ("request_obj", "code"),
    [
        ([], "bridge_bad_request"),
        ({}, "bridge_bad_request"),
        ({"repo": ""}, "bridge_bad_request"),
        ({"repo": "   "}, "bridge_bad_request"),
        ({"repo": "/x", "files_in_context": "src/a.ts"}, "bridge_bad_request"),
        ({"repo": "/x", "max_bytes": -1}, "bridge_bad_request"),
        ({"repo": "/x", "max_bytes": "lots"}, "bridge_bad_request"),
        ({"repo": "/x", "skill": ""}, "bridge_bad_request"),
        ({"repo": "/x", "home": ""}, "bridge_bad_request"),
    ],
)
def test_malformed_requests_are_refused_structurally(request_obj, code):
    response = skills_bridge.handle(request_obj)
    assert response["ok"] is False
    assert response["error"]["code"] == code


def test_a_missing_repository_is_reported_rather_than_looking_empty(tmp_path):
    response = skills_bridge.handle(
        {"repo": str(tmp_path / "nope"), "home": str(tmp_path / "no-home")},
    )
    # Still `ok` — a directory that does not exist is a discovery warning, not a bridge failure — but
    # the warning is present, so a host can tell "no skills" from "wrong path".
    assert response["ok"] is True
    assert response["skills"] == []
    assert any(w["code"] == "skills_root_missing" for w in response["warnings"])


def test_an_unreadable_user_layer_travels_to_the_host_with_its_errno(tmp_path):
    """The bridge's half of the defect: a layer it could not LIST must be named in the answer.

    MEASURED on 0.1.49: two launch shapes produced "8 skills" and "9 skills" for the same workspace and the
    same bridge, because `~/.codex` is a symlink into iCloud Drive and the LaunchServices-launched process had
    no TCC grant to list it. The answer carried NO warning, so the panel could not say which layer was short.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    write(home, ".agents/skills/visible/SKILL.md", skill_md(description="reachable"))
    write(home, ".codex/skills/sealed/SKILL.md", skill_md(description="unreachable"))
    sealed = home / ".codex" / "skills"
    # PROVEN, not assumed: skips with the reason when the mode is not enforced (root, or a filesystem without
    # POSIX mode bits), so this can never pass vacuously or fail for a reason unrelated to the bridge.
    with unlistable_directory(sealed):
        response = skills_bridge.handle({"repo": str(repo), "home": str(home)})

    assert response["ok"] is True
    assert [skill["name"] for skill in response["skills"]] == ["visible"]
    unreadable = [w for w in response["warnings"] if w["code"] == "skills_dir_unreadable"]
    assert len(unreadable) == 1, [w["code"] for w in response["warnings"]]
    assert ".codex/skills" in unreadable[0]["source_path"]
    assert unreadable[0]["message"]


# ---------------------------------------------------------------------------------------------
# main(): the process contract
# ---------------------------------------------------------------------------------------------

def _run(request_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "codna.skills_bridge"],
        input=request_text,
        text=True,
        capture_output=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PATH": "/usr/bin:/bin"},
        timeout=60,
    )


def test_main_writes_one_json_object_and_exits_zero():
    completed = _run(json.dumps({"repo": str(FIXTURE), "home": EMPTY_HOME}))
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert [skill["name"] for skill in payload["skills"]] == ["release", "pdf", "api"]


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
