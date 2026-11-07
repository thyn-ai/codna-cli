"""The fixture repository — the demonstration, pinned as a regression test.

``cli/tests/fixtures/rules_repo`` is a real, deliberately-shaped monorepo (nested rules, a globbed
rule, a catalogued rule, a manual rule, a cross-scope name conflict). The demonstration the PR reports
is the output of ``codna rules`` over THIS tree; these tests assert the properties that output claims,
so the demo cannot rot into a stale screenshot.

Every assertion here is about the RELATIONSHIP between a file set and the rules selected for it — the
thing two readers used to disagree about — not about a string in the renderer.
"""
from __future__ import annotations

from pathlib import Path

from codna.rules import load, request_rule

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rules_repo"


def product_roots(files) -> list[str]:
    """The repository-relative rule paths attached for ``files`` (no user layer: the fixture is hermetic)."""
    return [rule.source_path for rule in load(str(FIXTURE), files_in_context=files).result.kept]


def attached_names(files) -> list[str]:
    return [rule.name for rule in load(str(FIXTURE), files_in_context=files).result.kept]


def test_fixture_repo_exists():
    assert (FIXTURE / "AGENTS.md").is_file()
    assert (FIXTURE / ".codna" / "rules" / "python.mdc").is_file()


# ---------------------------------------------------------------------------------------------
# the file set decides
# ---------------------------------------------------------------------------------------------

def test_a_tsx_file_pulls_the_typescript_and_tsx_rules():
    attached = product_roots(["src/app.tsx"])
    assert ".codna/rules/typescript.mdc" in attached
    assert ".cursor/rules/no-default-exports.mdc" in attached
    # And NOT the CSS rule, which shares no extension with the file.
    assert ".codna/rules/styles.mdc" not in attached


def test_a_css_file_pulls_the_css_rule_and_not_the_typescript_one():
    attached = product_roots(["src/app.css"])
    assert ".codna/rules/styles.mdc" in attached
    assert ".codna/rules/typescript.mdc" not in attached
    assert ".cursor/rules/no-default-exports.mdc" not in attached


def test_a_docs_file_pulls_neither_language_rule():
    attached = product_roots(["docs/a.md"])
    assert ".codna/rules/typescript.mdc" not in attached
    assert ".codna/rules/styles.mdc" not in attached
    # The always-apply baseline is still there: scoping narrows, it does not switch the repo off.
    assert ".codna/rules/python.mdc" in attached


def test_an_api_python_file_pulls_the_nested_api_rule():
    attached = product_roots(["api/handlers/users.py"])
    assert "api/.codna/rules/pydantic.mdc" in attached
    # The nested rule is scoped to api/, so a file elsewhere must not pay for it.
    assert "api/.codna/rules/pydantic.mdc" not in product_roots(["src/app.tsx"])


def test_a_web_tsx_file_pulls_the_nested_react_rule_only_under_web():
    assert "web/.cursor/rules/react.mdc" in product_roots(["web/src/App.tsx"])
    assert "web/.cursor/rules/react.mdc" not in product_roots(["src/App.tsx"])


# ---------------------------------------------------------------------------------------------
# the four modes, on real files
# ---------------------------------------------------------------------------------------------

def test_the_fixture_covers_all_four_modes():
    from codna.rules import RuleMode

    modes = {rule.mode for rule in load(str(FIXTURE)).discovery.rules}
    assert modes == set(RuleMode)


def test_the_catalogued_rule_is_a_line_not_a_body_and_is_pullable():
    result = load(str(FIXTURE), files_in_context=["src/app.tsx"]).result
    assert "Migrations: create files under" not in result.text
    assert "migrations: How to write and run a database migration" in result.text

    pulled = request_rule(str(FIXTURE), "migrations")
    assert pulled is not None
    assert "make migrate" in pulled.body


def test_the_manual_rule_applies_only_when_mentioned():
    assert "Release: bump the version" not in (
        load(str(FIXTURE), files_in_context=["src/app.tsx"]).result.text)
    mentioned = load(str(FIXTURE), explicit_mentions=["@release"]).result.text
    assert "Release: bump the version" in mentioned


# ---------------------------------------------------------------------------------------------
# precedence and nesting, on real files
# ---------------------------------------------------------------------------------------------

def test_the_team_python_rule_beats_the_imported_cursor_one():
    result = load(str(FIXTURE), files_in_context=["src/app.py"]).result
    assert "Python: use ruff for lint and format" in result.text
    assert "PROJECT-SCOPE PYTHON RULE" not in result.text
    assert [d.source_path for d in result.dropped_for_precedence] == [".cursor/rules/python.mdc"]


def test_both_agents_files_apply_under_web_because_nesting_is_additive():
    text = load(str(FIXTURE), files_in_context=["web/src/App.tsx"]).result.text
    assert "Components live in" in text           # web/AGENTS.md
    assert "Run `pytest -q`" in text               # root AGENTS.md
    # Neither AGENTS.md counts as shadowing the other. (The `python` rule IS legitimately shadowed —
    # that conflict is in the same subtree — so assert on the AGENTS pair specifically.)
    shadowed = load(str(FIXTURE), files_in_context=["web/src/App.tsx"]).result.dropped_for_precedence
    assert all(dropped.name != "AGENTS" for dropped in shadowed)


def test_render_order_is_deterministic_and_byte_identical_across_runs():
    first = load(str(FIXTURE), files_in_context=["web/src/App.tsx"])
    second = load(str(FIXTURE), files_in_context=["web/src/App.tsx"])
    assert first.result.text == second.result.text
    assert [r.source_path for r in first.result.kept] == [r.source_path for r in second.result.kept]


def test_the_fixture_fits_a_normal_budget_so_the_demo_shows_no_drops():
    result = load(str(FIXTURE), files_in_context=["web/src/App.tsx"]).result
    assert result.dropped_for_budget == ()
    assert result.text


def test_a_tight_budget_reports_what_it_left_out():
    result = load(str(FIXTURE), files_in_context=["web/src/App.tsx"], max_bytes=400).result
    assert result.used_bytes <= 400
    assert result.dropped_for_budget, "the fixture must overflow a 400-byte budget"
    for dropped in result.dropped_for_budget:
        assert dropped.name and dropped.source_path and dropped.bytes_needed > 0
