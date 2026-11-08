"""``codna skills`` — the inspection surface, and the proof the engine's answer is real.

These are end-to-end tests over a real temporary repository: files are written, the CLI is invoked the
way a user invokes it, and the output is asserted. That is deliberate. A unit test can prove
``skill_catalogue`` returns an object; only this can prove the command a user runs reports the same
thing the prompt path does — which is the whole point of routing both through one engine.
"""
from __future__ import annotations

import json

import pytest

from codna import cli, skills_cli


@pytest.fixture(autouse=True)
def _hermetic_home(tmp_path, monkeypatch):
    """Point HOME at an empty directory for every test in this file.

    The command reads `~/.cursor/skills` and friends in production — that is the parity behaviour — so
    an assertion about "the skills in this repository" must not depend on what the machine running the
    suite happens to have installed in its own home.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


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


def _args(**kw):
    base = dict(
        repo=".", files=None, mention=None, budget=None, skill=None, strict=False, as_json=False,
    )
    base.update(kw)
    return type("Args", (), base)()


# ---------------------------------------------------------------------------------------------
# the command is reachable the way a user types it
# ---------------------------------------------------------------------------------------------

def test_skills_subcommand_is_registered():
    parser = cli.build_parser()
    args = parser.parse_args(["skills", ".", "--files", "src/a.ts", "--mention", "/pdf"])
    assert args.cmd == "skills"
    assert args.files == ["src/a.ts"]
    assert args.mention == ["/pdf"]


def test_skills_is_the_sibling_of_rules_not_a_flag_on_it():
    """Two commands, because they answer two questions — and a user must not have to guess which."""
    parser = cli.build_parser()
    assert parser.parse_args(["rules", "."]).cmd == "rules"
    assert parser.parse_args(["skills", "."]).cmd == "skills"


# ---------------------------------------------------------------------------------------------
# the human output
# ---------------------------------------------------------------------------------------------

def test_the_command_names_every_skill_and_its_state(tmp_path, capsys):
    write(tmp_path, ".cursor/skills/pdf/SKILL.md", skill_md(description="fill a form", paths="src/**/*.ts"))
    write(tmp_path, ".codna/skills/release/SKILL.md", skill_md(description="cut a release", manual=True))

    code = skills_cli.run_skills(_args(repo=str(tmp_path), files=["docs/a.md"]))
    out = capsys.readouterr().out

    assert code == 0
    assert "CATALOGUED" in out
    assert ".cursor/skills/pdf/SKILL.md" in out
    assert "paths mismatch" in out  # the scoped skill, honestly reported for this file set
    assert "manual" in out
    assert "codna_load_skill" in out  # the catalogue names the pull


def test_a_docs_context_and_a_code_context_report_different_catalogues(tmp_path, capsys):
    write(tmp_path, ".cursor/skills/pdf/SKILL.md", skill_md(description="pdf", paths="src/**/*.ts"))

    skills_cli.run_skills(_args(repo=str(tmp_path), files=["src/app.ts"]))
    code_out = capsys.readouterr().out
    skills_cli.run_skills(_args(repo=str(tmp_path), files=["docs/a.md"]))
    docs_out = capsys.readouterr().out

    assert "- pdf: pdf — paths: src/**/*.ts" in code_out
    assert "- pdf:" not in docs_out


def test_the_empty_answer_says_where_it_looked(tmp_path, capsys):
    code = skills_cli.run_skills(_args(repo=str(tmp_path)))
    out = capsys.readouterr().out
    assert code == 0
    assert "no skills found" in out
    assert ".cursor/skills" in out  # names the directories, so "nothing found" is actionable


def test_an_empty_answer_is_not_an_error(tmp_path):
    """A repository with no skills is the normal case, not a failure."""
    assert skills_cli.run_skills(_args(repo=str(tmp_path))) == 0


# ---------------------------------------------------------------------------------------------
# the pull a user types by hand
# ---------------------------------------------------------------------------------------------

def test_pulling_one_skill_prints_its_body_and_bundled_files(tmp_path, capsys):
    write(
        tmp_path,
        ".cursor/skills/pdf/SKILL.md",
        skill_md(description="pdf", body="USE THE SCRIPTS"),
    )
    write(tmp_path, ".cursor/skills/pdf/scripts/fill.ts", "export const x = 1;\n")

    code = skills_cli.run_skills(_args(repo=str(tmp_path), skill="pdf"))
    out = capsys.readouterr().out

    assert code == 0
    assert "USE THE SCRIPTS" in out
    assert "scripts/fill.ts" in out
    assert ".cursor/skills/pdf/SKILL.md" in out


def test_pulling_an_unknown_skill_refuses_with_a_code(tmp_path, capsys):
    code = skills_cli.run_skills(_args(repo=str(tmp_path), skill="nope"))
    err = capsys.readouterr().err
    assert code == 1
    payload = json.loads(err)
    assert payload["error"]["code"] == "skill_not_found"
    assert payload["error"]["details"]["requested"] == "nope"


# ---------------------------------------------------------------------------------------------
# JSON: the same answer, structured
# ---------------------------------------------------------------------------------------------

def test_json_reports_the_catalogue_the_drops_and_the_bodies_separately(tmp_path, capsys):
    for index in range(30):
        write(
            tmp_path,
            f".cursor/skills/s{index:02d}/SKILL.md",
            skill_md(description=f"skill number {index}", body="x" * 500),
        )

    skills_cli.run_skills(_args(repo=str(tmp_path), budget=600, as_json=True))
    payload = json.loads(capsys.readouterr().out)

    assert payload["used_bytes"] <= payload["budget_bytes"] == 600
    assert payload["discovered"]
    assert payload["catalogued"]
    assert payload["dropped"]
    assert len(payload["catalogued"]) + len(payload["dropped"]) == 30
    # The body is measured, never included — JSON is the catalogue's structured form, not the pull.
    assert all("body" not in skill for skill in payload["discovered"])
    assert all(skill["bytes"] > 0 for skill in payload["discovered"])
    assert "x" * 100 not in payload["text"]


def test_json_pull_includes_the_body(tmp_path, capsys):
    write(tmp_path, ".cursor/skills/pdf/SKILL.md", skill_md(description="pdf", body="REAL BODY"))
    skills_cli.run_skills(_args(repo=str(tmp_path), skill="pdf", as_json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["body"] == "REAL BODY"
    assert payload["name"] == "pdf"


def test_a_negative_budget_is_refused(tmp_path, capsys):
    code = skills_cli.run_skills(_args(repo=str(tmp_path), budget=-1))
    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "skills_budget_invalid"


def test_strict_turns_a_drop_into_a_failure_for_ci(tmp_path, capsys):
    for index in range(12):
        write(
            tmp_path,
            f".cursor/skills/s{index}/SKILL.md",
            skill_md(description=f"skill number {index} with a long description here"),
        )
    assert skills_cli.run_skills(_args(repo=str(tmp_path), budget=200)) == 0
    assert skills_cli.run_skills(_args(repo=str(tmp_path), budget=200, strict=True)) == 1


def test_a_drop_is_written_to_stderr_even_in_json_mode(tmp_path, capsys):
    """A machine consumer that ignores the JSON account must still leave a trace a human sees."""
    for index in range(12):
        write(
            tmp_path,
            f".cursor/skills/s{index}/SKILL.md",
            skill_md(description=f"skill number {index} with a long description here"),
        )
    skills_cli.run_skills(_args(repo=str(tmp_path), budget=200, as_json=True))
    err = capsys.readouterr().err
    assert "exceeded the 200-byte catalogue budget" in err


def test_the_budget_is_bounded_not_a_slice(tmp_path, capsys):
    """A budget that fits the header but no lines emits nothing and names every entry."""
    write(tmp_path, ".cursor/skills/x/SKILL.md", skill_md(description="d"))
    skills_cli.run_skills(_args(repo=str(tmp_path), budget=10, as_json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == ""
    assert payload["catalogued"] == []
    assert [entry["name"] for entry in payload["dropped"]] == ["x"]
