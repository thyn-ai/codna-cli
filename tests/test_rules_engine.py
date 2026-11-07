"""The rule engine: discovery, the four modes, precedence, globs, nesting and the budget.

Every test here is written against the PUBLIC surface (``codna.rules``) or the thin compatibility shim
(``codna.project_rules``), never against an internal helper, so the tests describe the behaviour a
consumer gets rather than the shape of the implementation.

The five behaviours the migration must not lose are named exactly as the request names them:

  * ``test_globs_scopes_to_matching_file``
  * ``test_four_modes``
  * ``test_precedence_team_project_user``
  * ``test_budget_reports_drops``
  * ``test_legacy_file_remains_always_on``
"""
from __future__ import annotations

import io

import pytest

from codna import project_rules
from codna.rules import (
    Rule,
    RuleMode,
    Scope,
    discover,
    guidance_for_repo,
    load,
    request_rule,
    select,
)
from codna.rules import globs as globs_mod
from codna.rules.render import DEFAULT_MAX_BYTES, render

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def write(root, rel: str, text: str, encoding: str = "utf-8"):
    """Create ``rel`` under ``root`` (parents included) and return its Path."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)
    return path


def mdc(description: str = "", globs: str = "", always: bool | None = None, body: str = "BODY") -> str:
    """A ``.mdc`` file with exactly the frontmatter keys asked for."""
    lines = ["---"]
    if description:
        lines.append(f"description: {description}")
    if globs:
        lines.append(f"globs: {globs}")
    if always is not None:
        lines.append(f"alwaysApply: {'true' if always else 'false'}")
    lines.append("---")
    return "\n".join(lines) + "\n" + body + "\n"


def names(rules) -> list[str]:
    return [rule.name for rule in rules]


def by_name(rules, name: str) -> Rule:
    for rule in rules:
        if rule.name == name:
            return rule
    raise AssertionError(f"{name!r} not among {names(rules)}")


# ---------------------------------------------------------------------------------------------
# globs — the matcher the scoping depends on
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("src/**/*.ts", "src/a.ts", True),            # ** crosses ZERO directories
        ("src/**/*.ts", "src/deep/nested/a.ts", True),
        ("src/*.ts", "src/a.ts", True),
        ("src/*.ts", "src/deep/a.ts", False),          # * never crosses a separator
        ("**/*.tsx", "a.tsx", True),
        ("**/*.tsx", "src/ui/a.tsx", True),
        ("docs/**", "docs/a.md", True),
        ("docs/**", "docs/a/b.md", True),
        ("docs/**", "src/a.md", False),
        ("{python,rust}/**", "python/x.py", True),
        ("{python,rust}/**", "ruby/x.rb", False),
        ("**/*.{ts,tsx}", "src/a.tsx", True),          # brace + nested ** together
        ("**/*.{ts,tsx}", "src/a.py", False),
        ("a/**/b/c.py", "a/b/c.py", True),             # interior ** matches zero segments
        ("a/**/b/c.py", "a/x/y/b/c.py", True),
        ("Makefile", "Makefile", True),
        ("Makefile", "sub/Makefile", False),
    ],
)
def test_glob_semantics(pattern, path, expected):
    assert globs_mod.matches(pattern, path) is expected


def test_comma_separated_globs_are_all_kept():
    """``globs: a, b`` is a documented shape; losing the second pattern is the silent-drop defect."""
    assert globs_mod.parse_globs("src/**/*.ts, tests/**") == ("src/**/*.ts", "tests/**")
    # A comma inside a brace group belongs to the group, not to the list.
    assert globs_mod.parse_globs("{a,b},c") == ("{a,b}", "c")


def test_globs_accept_a_yaml_list():
    assert globs_mod.parse_globs(["src/**", "tests/**"]) == ("src/**", "tests/**")


def test_globs_reject_a_non_string():
    with pytest.raises(ValueError):
        globs_mod.parse_globs(42)


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_globs_scopes_to_matching_file
# ---------------------------------------------------------------------------------------------

def test_globs_scopes_to_matching_file(tmp_path):
    """Two rules, one globbed: absent for ``docs/a.md``, present for ``src/a.ts``."""
    write(tmp_path, ".codna/rules/general.mdc", mdc(always=True, body="GENERAL RULE"))
    write(tmp_path, ".codna/rules/ts.mdc", mdc(globs="src/**/*.ts", body="TS RULE"))

    for_docs = guidance_for_repo(str(tmp_path), files_in_context=["docs/a.md"])
    assert "GENERAL RULE" in for_docs
    assert "TS RULE" not in for_docs

    for_src = guidance_for_repo(str(tmp_path), files_in_context=["src/a.ts"])
    assert "GENERAL RULE" in for_src
    assert "TS RULE" in for_src

    # And the same question through the structured API, so the reason is observable, not inferred.
    loaded = load(str(tmp_path), files_in_context=["docs/a.md"])
    assert names(loaded.result.kept) == ["general"]
    assert names(loaded.selection.unmatched) == ["ts"]


def test_globs_at_depth_use_the_repository_relative_path(tmp_path):
    """A glob is matched against the path as the user writes it, not as a bare filename."""
    write(tmp_path, ".codna/rules/x.mdc", mdc(globs="src/**/*.ts"))
    assert "x" not in (guidance_for_repo(str(tmp_path), files_in_context=["a.ts"]) or "")
    assert "x" in (guidance_for_repo(str(tmp_path), files_in_context=["src/a.ts"]) or "")


def test_no_files_named_includes_glob_rules_and_reports_them(tmp_path):
    """An UNKNOWN file set cannot rule a glob rule out, so it is included and reported.

    This is the durable-run case: its agent may edit anything, so withholding a globbed convention
    would be silent under-application — the agent would not be told a rule governing the files it is
    about to change. ``unmatched`` keeps the fact visible instead.
    """
    write(tmp_path, ".codna/rules/general.mdc", mdc(always=True, body="GENERAL"))
    write(tmp_path, ".codna/rules/ts.mdc", mdc(globs="src/**/*.ts", body="TS"))
    selection = load(str(tmp_path)).selection
    assert names(selection.selected) == ["general", "ts"]
    assert selection.files_known is False
    assert names(selection.unmatched) == ["ts"]


def test_an_explicitly_empty_file_set_excludes_glob_rules(tmp_path):
    """A KNOWN empty set (nothing changed) matches nothing, which is a different answer."""
    write(tmp_path, ".codna/rules/general.mdc", mdc(always=True, body="GENERAL"))
    write(tmp_path, ".codna/rules/ts.mdc", mdc(globs="src/**/*.ts", body="TS"))
    selection = load(str(tmp_path), files_in_context=[]).selection
    assert names(selection.selected) == ["general"]
    assert selection.files_known is True
    assert names(selection.unmatched) == ["ts"]


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_four_modes
# ---------------------------------------------------------------------------------------------

def test_four_modes(tmp_path):
    """One fixture per row of the mode interaction table, checked in both directions."""
    write(tmp_path, ".codna/rules/always.mdc", mdc(always=True, body="ALWAYS BODY"))
    write(tmp_path, ".codna/rules/scoped.mdc", mdc(globs="src/**/*.py", body="SCOPED BODY"))
    write(tmp_path, ".codna/rules/described.mdc",
          mdc(description="how to run the db migrations", body="DESCRIBED BODY"))
    write(tmp_path, ".codna/rules/manual.mdc", mdc(body="MANUAL BODY"))

    rules = discover(str(tmp_path)).rules
    assert {rule.name: rule.mode for rule in rules} == {
        "always": RuleMode.ALWAYS,
        "scoped": RuleMode.GLOB,
        "described": RuleMode.CATALOG,
        "manual": RuleMode.MANUAL,
    }

    # -- 1. ALWAYS: present with no files at all, and with unrelated files.
    for files in (None, [], ["docs/a.md"], ["src/a.py"]):
        assert "ALWAYS BODY" in (guidance_for_repo(str(tmp_path), files_in_context=files) or "")

    # -- 2. GLOB: present only when a matching file is in context.
    assert "SCOPED BODY" in (guidance_for_repo(str(tmp_path), files_in_context=["src/a.py"]) or "")
    assert "SCOPED BODY" not in (guidance_for_repo(str(tmp_path), files_in_context=["docs/a.md"]) or "")

    # -- 3. CATALOG: never rendered as a body; one line that names it and says how to pull it.
    rendered = guidance_for_repo(str(tmp_path), files_in_context=["src/a.py"]) or ""
    assert "DESCRIBED BODY" not in rendered
    assert "described" in rendered
    assert "how to run the db migrations" in rendered
    assert "codna_request_rule" in rendered

    # ... and the pull mechanism actually returns it, by the name the catalog printed.
    pulled = request_rule(str(tmp_path), "described")
    assert pulled is not None and pulled.body == "DESCRIBED BODY"

    # -- 4. MANUAL: absent by default, present when explicitly mentioned.
    assert "MANUAL BODY" not in rendered
    mentioned = guidance_for_repo(str(tmp_path), explicit_mentions=["@manual"])
    assert "MANUAL BODY" in mentioned


def test_catalog_rule_body_is_pulled_only_by_name(tmp_path):
    """A description-only rule is a CATALOG entry, and its body is reachable by the printed name."""
    write(tmp_path, ".codna/rules/db.mdc", mdc(description="database conventions", body="USE MIGRATIONS"))
    assert request_rule(str(tmp_path), "db").body == "USE MIGRATIONS"
    # `@name` and the bare filename both resolve — a user typing either must not get nothing.
    assert request_rule(str(tmp_path), "@db").body == "USE MIGRATIONS"
    assert request_rule(str(tmp_path), "db.mdc").body == "USE MIGRATIONS"
    assert request_rule(str(tmp_path), "not-a-rule") is None


def test_mentioning_a_catalog_rule_promotes_it_to_a_full_body(tmp_path):
    write(tmp_path, ".codna/rules/db.mdc", mdc(description="database conventions", body="USE MIGRATIONS"))
    text = guidance_for_repo(str(tmp_path), explicit_mentions=["@db"]) or ""
    assert "USE MIGRATIONS" in text


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_precedence_team_project_user
# ---------------------------------------------------------------------------------------------

def test_precedence_team_project_user(tmp_path, tmp_path_factory):
    """Conflicting bodies for the SAME rule name; the winner is by SOURCE."""
    write(tmp_path, ".codna/rules/style.mdc", mdc(always=True, body="TEAM BODY"))
    write(tmp_path, ".cursor/rules/style.mdc", mdc(always=True, body="PROJECT BODY"))
    user_dir = tmp_path_factory.mktemp("home") / "rules"
    write(user_dir, "style.mdc", mdc(always=True, body="USER BODY"))

    loaded = load(str(tmp_path), user_rules_dir=str(user_dir))

    assert "TEAM BODY" in loaded.result.text
    assert "PROJECT BODY" not in loaded.result.text
    assert "USER BODY" not in loaded.result.text
    assert [d.reason for d in loaded.result.dropped_for_precedence] == ["precedence", "precedence"]


def test_precedence_project_beats_user_when_no_team_rule(tmp_path, tmp_path_factory):
    write(tmp_path, ".cursor/rules/style.mdc", mdc(always=True, body="PROJECT BODY"))
    user_dir = tmp_path_factory.mktemp("home") / "rules"
    write(user_dir, "style.mdc", mdc(always=True, body="USER BODY"))
    loaded = load(str(tmp_path), user_rules_dir=str(user_dir))
    assert "PROJECT BODY" in loaded.result.text
    assert "USER BODY" not in loaded.result.text


def test_precedence_is_decided_by_source_not_by_input_order():
    """``select`` sorts by scope, so a caller's list order cannot decide the winner.

    This is the unit-level statement of the property: the winner is a function of ``Rule.scope``, not
    of where the rule happened to appear in the list the engine was handed.
    """
    team = Rule(name="style", body="TEAM", source_path=".codna/rules/style.mdc", scope=Scope.TEAM,
                always_apply=True)
    project = Rule(name="style", body="PROJECT", source_path=".cursor/rules/style.mdc",
                    scope=Scope.PROJECT, always_apply=True)
    user = Rule(name="style", body="USER", source_path="~/rules/style.mdc", scope=Scope.USER,
                always_apply=True)

    for ordering in ((project, user, team), (user, project, team), (team, user, project)):
        result = render(select(ordering))
        assert "TEAM" in result.text, ordering
        assert "PROJECT" not in result.text
        assert "USER" not in result.text
        assert len(result.dropped_for_precedence) == 2


def test_non_conflicting_rules_all_survive_across_scopes(tmp_path, tmp_path_factory):
    write(tmp_path, ".codna/rules/team.mdc", mdc(always=True, body="TEAM"))
    write(tmp_path, ".cursor/rules/proj.mdc", mdc(always=True, body="PROJECT"))
    user_dir = tmp_path_factory.mktemp("home") / "rules"
    write(user_dir, "user.mdc", mdc(always=True, body="USER"))
    text = load(str(tmp_path), user_rules_dir=str(user_dir)).result.text
    assert "TEAM" in text and "PROJECT" in text and "USER" in text


def test_render_order_is_team_then_project_then_user(tmp_path, tmp_path_factory):
    write(tmp_path, ".codna/rules/t.mdc", mdc(always=True, body="T"))
    write(tmp_path, ".cursor/rules/p.mdc", mdc(always=True, body="P"))
    user_dir = tmp_path_factory.mktemp("home") / "rules"
    write(user_dir, "u.mdc", mdc(always=True, body="U"))
    loaded = load(str(tmp_path), user_rules_dir=str(user_dir))
    assert [rule.scope for rule in loaded.result.kept] == [
        Scope.TEAM, Scope.PROJECT, Scope.USER,
    ]


# ---------------------------------------------------------------------------------------------
# nesting
# ---------------------------------------------------------------------------------------------

def test_nested_rule_applies_only_to_its_subtree(tmp_path):
    write(tmp_path, "pkg/api/.codna/rules/api.mdc", mdc(always=True, body="API RULE"))
    write(tmp_path, ".codna/rules/root_rule.mdc", mdc(always=True, body="ROOT RULE"))

    inside = guidance_for_repo(str(tmp_path), files_in_context=["pkg/api/x.py"]) or ""
    outside = guidance_for_repo(str(tmp_path), files_in_context=["pkg/web/x.py"]) or ""
    assert "API RULE" in inside
    assert "ROOT RULE" in inside
    assert "API RULE" not in outside
    assert "ROOT RULE" in outside


def test_nested_agents_md_applies_only_to_its_subtree(tmp_path):
    write(tmp_path, "pkg/api/AGENTS.md", "# API\nUse dataclasses.\n")
    inside = guidance_for_repo(str(tmp_path), files_in_context=["pkg/api/x.py"]) or ""
    outside = guidance_for_repo(str(tmp_path), files_in_context=["pkg/web/x.py"]) or ""
    assert "Use dataclasses." in inside
    assert "Use dataclasses." not in outside


def test_nested_rule_is_reported_when_out_of_scope(tmp_path):
    write(tmp_path, "pkg/api/.codna/rules/api.mdc", mdc(always=True, body="API RULE"))
    loaded = load(str(tmp_path), files_in_context=["pkg/web/x.py"])
    assert names(loaded.selection.out_of_scope) == ["api"]
    assert names(loaded.result.kept) == []


def test_nested_rules_sort_most_specific_first_within_a_scope(tmp_path):
    write(tmp_path, ".codna/rules/root.mdc", mdc(always=True, body="ROOT"))
    write(tmp_path, "pkg/.codna/rules/mid.mdc", mdc(always=True, body="MID"))
    write(tmp_path, "pkg/api/.codna/rules/leaf.mdc", mdc(always=True, body="LEAF"))
    loaded = load(str(tmp_path), files_in_context=["pkg/api/x.py"])
    assert names(loaded.result.kept) == ["leaf", "mid", "root"]


def test_nested_rules_are_additive_not_conflicting(tmp_path):
    """A nested rule refining a root rule with the same name must NOT shadow it.

    ``web/AGENTS.md`` and the root ``AGENTS.md`` share a filename — and therefore a default rule name —
    but they govern different subtrees, so both must apply to a file under ``web/``. Treating them as a
    name conflict silently dropped one, which is the exact failure mode precedence must not introduce.
    """
    write(tmp_path, "AGENTS.md", "ROOT AGENTS RULE\n")
    write(tmp_path, "web/AGENTS.md", "WEB AGENTS RULE\n")
    loaded = load(str(tmp_path), files_in_context=["web/src/App.tsx"])
    assert "ROOT AGENTS RULE" in loaded.result.text
    assert "WEB AGENTS RULE" in loaded.result.text
    assert loaded.result.dropped_for_precedence == ()


def test_nested_rule_with_the_same_name_shadows_only_within_its_own_subtree(tmp_path):
    """A genuine same-name conflict is resolved inside one subtree, leaving the other untouched."""
    write(tmp_path, ".codna/rules/db.mdc", mdc(always=True, body="ROOT DB"))
    write(tmp_path, "pkg/.cursor/rules/db.mdc", mdc(always=True, body="NESTED DB"))
    loaded = load(str(tmp_path), files_in_context=["pkg/x.py"])
    # Different subtrees -> both apply, and this is not a precedence drop.
    assert "ROOT DB" in loaded.result.text
    assert "NESTED DB" in loaded.result.text
    assert loaded.result.dropped_for_precedence == ()


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_budget_reports_drops
# ---------------------------------------------------------------------------------------------

def test_budget_reports_drops(tmp_path):
    """40 rules over budget: the kept set is deterministic and the dropped names are RETURNED."""
    for index in range(40):
        write(tmp_path, f".codna/rules/r{index:02d}.mdc",
              mdc(always=True, body=f"RULE-{index:02d} " + "x" * 200))

    budget = 2_000
    first = load(str(tmp_path), max_bytes=budget).result
    second = load(str(tmp_path), max_bytes=budget).result

    # Deterministic: the same tree and budget give the same kept set, byte for byte.
    assert [rule.name for rule in first.kept] == [rule.name for rule in second.kept]
    assert first.text == second.text
    assert first.used_bytes <= budget

    # Nothing vanished: every rule is either kept or named as dropped, exactly once.
    kept = {rule.name for rule in first.kept}
    dropped = [d for d in first.dropped_for_budget]
    assert len(dropped) > 0, "the fixture must overflow the budget for this test to mean anything"
    assert kept | {d.name for d in dropped} == {f"r{i:02d}" for i in range(40)}
    assert not (kept & {d.name for d in dropped})
    assert all(d.bytes_needed > 0 for d in dropped)


def test_budget_does_not_starve_a_small_rule_behind_a_big_one(tmp_path):
    """A rule too big for the remaining budget is skipped, and the scan CONTINUES."""
    write(tmp_path, ".codna/rules/a-big.mdc", mdc(always=True, body="BIG " + "x" * 3000))
    write(tmp_path, ".codna/rules/b-small.mdc", mdc(always=True, body="SMALL RULE"))
    result = load(str(tmp_path), max_bytes=1_500).result
    # Sorted order is a-big then b-small; a-big cannot fit at all, so b-small must still render.
    assert names(result.kept) == ["b-small"]
    assert "SMALL RULE" in result.text
    assert [d.name for d in result.dropped_for_budget] == ["a-big"]


def test_budget_drops_are_written_to_stderr(tmp_path):
    """The reporting half: a drop must reach a stream a user reads, not just a return value."""
    for index in range(10):
        write(tmp_path, f".codna/rules/r{index}.mdc", mdc(always=True, body="y" * 400))
    stream = io.StringIO()
    guidance_for_repo(str(tmp_path), max_bytes=1_000, stream=stream)
    logged = stream.getvalue()
    assert "exceeded the 1000-byte budget" in logged
    assert "were NOT attached" in logged


def test_a_single_oversized_rule_is_truncated_reported_not_dropped(tmp_path):
    """A repository with one huge rule must still get guidance — cut, but announced.

    This is the exact case the old reader handled by slicing, and the durable-path test pins it: a
    16 KB AGENTS.md must produce text, not an empty string.
    """
    write(tmp_path, "AGENTS.md", "z" * 40_000)
    stream = io.StringIO()
    text = guidance_for_repo(str(tmp_path), max_bytes=4_000, stream=stream)
    assert text and len(text.encode("utf-8")) <= 4_000
    assert "truncated" in text
    assert "rule_truncated_for_budget" in stream.getvalue()


def test_budget_bound_is_byte_accurate_not_character_accurate(tmp_path):
    """Multi-byte content must not push the rendered prompt past the budget."""
    write(tmp_path, ".codna/rules/unicode.mdc", mdc(always=True, body="→" * 500))
    result = load(str(tmp_path), max_bytes=1_000).result
    assert result.used_bytes <= 1_000


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_legacy_file_remains_always_on
# ---------------------------------------------------------------------------------------------

def test_legacy_file_remains_always_on(tmp_path):
    """A ``.codna/rules`` FILE with no frontmatter still applies everywhere."""
    write(tmp_path, ".codna/rules", "Prefer small diffs.\n")
    for files in (None, [], ["docs/a.md"], ["src/deep/a.ts"]):
        text = guidance_for_repo(str(tmp_path), files_in_context=files) or ""
        assert "Prefer small diffs." in text, files


@pytest.mark.parametrize(
    ("rel", "marker"),
    [
        (".cursorrules", "legacy cursorrules"),
        ("BUGBOT.md", "legacy bugbot"),
        (".cursor/BUGBOT.md", "legacy cursor bugbot"),
        ("AGENTS.md", "legacy agents"),
        (".codna/review.md", "legacy review"),
        (".codna/rules.md", "legacy rules md"),
        (".cursor/rules.md", "legacy cursor rules md"),
    ],
)
def test_every_legacy_root_file_stays_always_on(tmp_path, rel, marker):
    """Every historical filename keeps its always-on meaning — no frontmatter, no scoping."""
    write(tmp_path, rel, marker + "\n")
    for files in (None, [], ["docs/a.md"], ["src/a.ts"]):
        assert marker in (guidance_for_repo(str(tmp_path), files_in_context=files) or ""), files


def test_a_legacy_file_can_opt_into_scoping_with_explicit_frontmatter(tmp_path):
    """Frontmatter is the opt-in: ``alwaysApply: false`` must be honoured, or the escape hatch is a lie."""
    write(tmp_path, ".cursorrules", mdc(always=False, globs="src/**/*.py", body="PY ONLY"))
    assert "PY ONLY" not in (guidance_for_repo(str(tmp_path), files_in_context=["docs/a.md"]) or "")
    assert "PY ONLY" in (guidance_for_repo(str(tmp_path), files_in_context=["src/a.py"]) or "")


def test_a_legacy_file_with_only_a_description_stays_always_on(tmp_path):
    """``description:`` alone must NOT demote an always-on file to a catalog line."""
    write(tmp_path, "AGENTS.md", mdc(description="project conventions", body="ALWAYS RULE"))
    text = guidance_for_repo(str(tmp_path), files_in_context=["docs/a.md"]) or ""
    assert "ALWAYS RULE" in text


def test_rules_directory_file_without_frontmatter_is_kept_and_warned(tmp_path):
    """A ``.mdc`` in a rules dir with no frontmatter stays effective, and says how to scope it."""
    write(tmp_path, ".codna/rules/bare.mdc", "no frontmatter here\n")
    loaded = load(str(tmp_path), files_in_context=["docs/a.md"])
    assert names(loaded.result.kept) == ["bare"]
    codes = {warning.code for warning in loaded.discovery.warnings}
    assert "rule_without_frontmatter" in codes


# ---------------------------------------------------------------------------------------------
# the compatibility shim
# ---------------------------------------------------------------------------------------------

def test_read_project_guidance_keeps_its_contract(tmp_path):
    """Existing callers: ``None`` when absent, a labelled string when present."""
    assert project_rules.read_project_guidance(str(tmp_path)) is None
    write(tmp_path, "AGENTS.md", "Test cmd: pytest -q\n")
    text = project_rules.read_project_guidance(str(tmp_path))
    assert text is not None and text.startswith("# AGENTS.md")
    assert "pytest -q" in text


def test_read_project_guidance_missing_directory_is_none():
    assert project_rules.read_project_guidance("/no/such/repo/at/all") is None


def test_shim_and_engine_agree(tmp_path):
    """Two entry points, one answer — the disagreement that motivated the engine is gone."""
    write(tmp_path, ".codna/rules/a.mdc", mdc(always=True, body="A"))
    write(tmp_path, ".codna/rules/b.mdc", mdc(globs="src/**/*.ts", body="B"))
    shim = project_rules.guidance_for_files(str(tmp_path), ["src/x.ts"])
    engine = guidance_for_repo(str(tmp_path), files_in_context=["src/x.ts"])
    assert shim == engine


def test_shim_legacy_file_list_is_pinned_to_the_engine(tmp_path):
    """``project_rules._RULES_FILES`` is documentation; it must not drift from what is read."""
    from codna.rules.discovery import _LEGACY_ROOT_FILES

    assert set(project_rules._RULES_FILES) == set(_LEGACY_ROOT_FILES)


def test_shim_max_bytes_is_the_engine_default():
    assert project_rules._MAX_BYTES == DEFAULT_MAX_BYTES


# ---------------------------------------------------------------------------------------------
# discovery robustness — failures are reported, never silently absorbed
# ---------------------------------------------------------------------------------------------

def test_malformed_frontmatter_is_reported_and_the_file_skipped(tmp_path):
    write(tmp_path, ".codna/rules/broken.mdc", "---\nglobs: [unclosed\n---\nbody\n")
    loaded = load(str(tmp_path))
    assert names(loaded.result.kept) == []
    assert any(w.code == "rule_frontmatter_invalid" for w in loaded.discovery.warnings)


def test_empty_body_is_reported(tmp_path):
    write(tmp_path, ".codna/rules/empty.mdc", "---\nalwaysApply: true\n---\n   \n")
    loaded = load(str(tmp_path))
    assert names(loaded.result.kept) == []
    assert any(w.code == "rule_body_empty" for w in loaded.discovery.warnings)


def test_missing_root_is_reported_not_raised():
    discovery = discover("/no/such/repo/anywhere")
    assert discovery.rules == ()
    assert [w.code for w in discovery.warnings] == ["rules_root_missing"]


def test_discovery_is_deterministic_regardless_of_creation_order(tmp_path):
    for name in ("c", "a", "b"):
        write(tmp_path, f".codna/rules/{name}.mdc", mdc(always=True, body=name.upper()))
    first = [r.name for r in discover(str(tmp_path)).rules]
    second = [r.name for r in discover(str(tmp_path)).rules]
    assert first == second == sorted(first)


def test_plain_md_inside_cursor_rules_is_read(tmp_path):
    """Deliberate divergence: Cursor ignores ``.md`` in ``.cursor/rules``; Codna reads it."""
    write(tmp_path, ".cursor/rules/conventions.md", mdc(always=True, body="CURSOR MD RULE"))
    assert "CURSOR MD RULE" in (guidance_for_repo(str(tmp_path), files_in_context=["a.py"]) or "")


def test_node_modules_and_git_are_not_walked(tmp_path):
    write(tmp_path, "node_modules/pkg/AGENTS.md", "SHOULD NOT APPEAR\n")
    write(tmp_path, ".git/AGENTS.md", "SHOULD NOT APPEAR\n")
    write(tmp_path, "AGENTS.md", "REAL RULE\n")
    text = guidance_for_repo(str(tmp_path)) or ""
    assert "REAL RULE" in text
    assert "SHOULD NOT APPEAR" not in text


# ---------------------------------------------------------------------------------------------
# select: purity
# ---------------------------------------------------------------------------------------------

def test_select_is_pure_and_total():
    rules = (
        Rule(name="a", body="A", source_path="a", scope=Scope.TEAM, always_apply=True),
        Rule(name="b", body="B", source_path="b", scope=Scope.PROJECT, globs=("src/**",)),
    )
    first = select(rules, ["src/x.py"])
    second = select(rules, ["src/x.py"])
    assert first == second
    assert names(first.selected) == ["a", "b"]
    # And no mutation: the input tuple is unchanged and the rules are the same objects.
    assert rules[0].always_apply is True


def test_select_normalises_paths_and_mentions():
    from codna.rules.select import normalise_mentions, normalise_paths

    assert normalise_paths(["./a/b.py", "a//b.py", "a/b.py", "", "/c.py"]) == ("a/b.py", "c.py")
    # A rule extension is stripped AND the literal name is kept (a rule could be named `python.mdc`).
    assert normalise_mentions(["python.mdc"]) == frozenset({"python", "python.mdc"})
    assert "@Python".casefold() and normalise_mentions(["@Python"]) == frozenset({"python"})
    # A DOTTED rule name keeps its dots: only rule extensions are stripped.
    assert normalise_mentions(["db.v2"]) == frozenset({"db.v2"})
