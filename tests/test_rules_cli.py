"""``codna rules`` — the inspection surface, and the proof the engine's answer is real.

These are end-to-end tests over a real temporary repository: files are written, the CLI is invoked the
way a user invokes it, and the output is asserted. That is deliberate. A unit test can prove ``select``
returns a tuple; only this can prove the command a user runs reports the same thing the prompt path
does, which is the whole point of routing both through one engine.
"""
from __future__ import annotations

import json

from codna import cli, rules_cli


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


def _args(**kw):
    base = dict(
        repo=".", files=None, mention=None, budget=None, rule=None, user_rules_dir=None,
        no_catalog=False, strict=False, as_json=False,
    )
    base.update(kw)
    return type("Args", (), base)()


# ---------------------------------------------------------------------------------------------
# the command is reachable the way a user types it
# ---------------------------------------------------------------------------------------------

def test_rules_subcommand_is_registered():
    parser = cli.build_parser()
    args = parser.parse_args(["rules", ".", "--files", "src/a.ts", "--mention", "@python"])
    assert args.cmd == "rules"
    assert args.files == ["src/a.ts"]
    assert args.mention == ["@python"]
    assert args.func is rules_cli.run_rules


def test_rules_is_in_the_help_metavar():
    """A command a user cannot find in ``--help`` is a command they do not have."""
    help_text = cli.build_parser().format_help()
    assert "rules" in help_text


# ---------------------------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------------------------

def test_cli_shows_scoped_rules_for_a_file_set(tmp_path, capsys):
    write(tmp_path, ".codna/rules/general.mdc", mdc(always=True, body="GENERAL"))
    write(tmp_path, ".codna/rules/ts.mdc", mdc(globs="src/**/*.ts", body="TS ONLY"))

    code = rules_cli.run_rules(_args(repo=str(tmp_path), files=["src/a.ts"]))
    out = capsys.readouterr().out
    assert code == 0
    assert "GENERAL" in out and "TS ONLY" in out
    assert "[ATTACHED" in out

    code = rules_cli.run_rules(_args(repo=str(tmp_path), files=["docs/a.md"]))
    out = capsys.readouterr().out
    assert code == 0
    assert "[ATTACHED" in out
    assert "[no glob match" in out
    assert "TS ONLY" not in out


def test_cli_json_is_the_structured_answer(tmp_path, capsys):
    write(tmp_path, ".codna/rules/a.mdc", mdc(always=True, body="A"))
    write(tmp_path, ".codna/rules/described.mdc", mdc(description="needs pulling", body="B"))
    code = rules_cli.run_rules(_args(repo=str(tmp_path), as_json=True))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [rule["name"] for rule in payload["attached"]] == ["a"]
    assert [rule["name"] for rule in payload["catalogued"]] == ["described"]
    assert payload["discovered"][0]["mode"] in {"always", "catalog"}


def test_cli_reports_dropped_rules_on_stderr(tmp_path, capsys):
    for index in range(12):
        write(tmp_path, f".codna/rules/r{index}.mdc", mdc(always=True, body="z" * 400))
    code = rules_cli.run_rules(_args(repo=str(tmp_path), budget=1_200, as_json=True))
    captured = capsys.readouterr()
    assert code == 0  # a drop is reported, not a failure
    assert "were NOT attached" in captured.err
    payload = json.loads(captured.out)
    assert payload["dropped"], "the JSON must carry the same account stderr printed"


def test_cli_strict_exits_one_on_a_drop(tmp_path, capsys):
    for index in range(12):
        write(tmp_path, f".codna/rules/r{index}.mdc", mdc(always=True, body="z" * 400))
    code = rules_cli.run_rules(_args(repo=str(tmp_path), budget=1_200, strict=True))
    captured = capsys.readouterr()
    assert code == 1
    assert "were NOT attached" in captured.err


def test_cli_prints_one_rule_body(tmp_path, capsys):
    write(tmp_path, ".codna/rules/db.mdc", mdc(description="database", body="USE MIGRATIONS"))
    code = rules_cli.run_rules(_args(repo=str(tmp_path), rule="db"))
    out = capsys.readouterr().out
    assert code == 0
    assert "USE MIGRATIONS" in out
    assert "mode: catalog" in out


def test_cli_reports_an_unknown_rule_name_structurally(tmp_path, capsys):
    write(tmp_path, ".codna/rules/db.mdc", mdc(always=True, body="X"))
    code = rules_cli.run_rules(_args(repo=str(tmp_path), rule="nope"))
    err = capsys.readouterr().err
    assert code == 1
    payload = json.loads(err)
    assert payload["error"]["code"] == "rule_not_found"


def test_cli_rejects_a_negative_budget(tmp_path, capsys):
    code = rules_cli.run_rules(_args(repo=str(tmp_path), budget=-1))
    payload = json.loads(capsys.readouterr().err)
    assert code == 1
    assert payload["error"]["code"] == "rules_budget_invalid"


def test_cli_works_through_main_dispatch(tmp_path, capsys):
    """The real entry point, not the handler: proves the wiring and the exit code together."""
    write(tmp_path, ".codna/rules/a.mdc", mdc(always=True, body="THROUGH MAIN"))
    code = cli.main(["rules", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "THROUGH MAIN" in out


def test_cli_on_a_repository_with_no_rules_says_so(tmp_path, capsys):
    code = rules_cli.run_rules(_args(repo=str(tmp_path)))
    out = capsys.readouterr().out
    assert code == 0
    assert "no rules found" in out
