"""The skill engine: discovery, ``paths`` scoping, manual-only, the bounded catalogue, and the pull.

Every test is written against the PUBLIC surface (``codna.rules``'s skill exports) or the CLI/bridge
that a consumer actually calls, never against an internal helper. The five behaviours the request
names are tested by name, at the top of each section:

  * ``test_cursor_skills_dir_discovered``
  * ``test_paths_scoping``
  * ``test_disable_model_invocation_is_manual_only``
  * ``test_catalogue_is_bounded``
  * ``test_skill_catalogue_reaches_session``   (in ``test_rules_bridge.py`` — see the note there)

The other property this file defends is the one that makes skills a *parity* fix rather than a new
feature: the engine REUSES the rule machinery. ``test_paths_scoping_reads_the_same_glob_language``
asserts that by comparing a skill's ``paths`` decision to a rule's ``globs`` decision on the same
pattern and the same file — if either matcher were reimplemented, the two would drift apart first and
this test second.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from codna.rules import (
    Skill,
    SkillMode,
    SkillScope,
    discover_skills,
    load_context,
    load_skill,
    skill_catalogue,
)
from codna.rules import globs as globs_mod
from codna.rules.render import CatalogueLine, render_catalogue
from codna.rules.skills import CATALOGUE_HEADER, MAX_SKILLS, search_dirs

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skills_repo"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def skill_md(
    *,
    name: str = "",
    description: str = "",
    paths: str = "",
    manual: bool | None = None,
    body: str = "BODY",
) -> str:
    """A ``SKILL.md`` with exactly the frontmatter keys asked for."""
    lines = ["---"]
    if name:
        lines.append(f"name: {name}")
    if description:
        lines.append(f"description: {description}")
    if paths:
        lines.append(f"paths: {paths}")
    if manual is not None:
        lines.append(f"disable-model-invocation: {'true' if manual else 'false'}")
    lines.append("---")
    return "\n".join(lines) + "\n" + body + "\n"


def names(skills) -> list[str]:
    return [skill.name for skill in skills]


def by_name(skills, name: str) -> Skill:
    for skill in skills:
        if skill.name == name:
            return skill
    raise AssertionError(f"{name!r} not among {names(skills)}")


def hermetic(root: Path):
    """Discovery with the USER layer pointed at a directory that does not exist.

    Every test here must be independent of whatever the machine running it happens to have in
    ``~/.cursor/skills`` — otherwise a green suite on one laptop is a red one on another.
    """
    return discover_skills(str(root), home=str(root / "no-home"), include_user=False)


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_cursor_skills_dir_discovered
# ---------------------------------------------------------------------------------------------

def test_cursor_skills_dir_discovered(tmp_path):
    """A ``.cursor/skills/x/SKILL.md`` is found — the directory neither reader used to read."""
    write(tmp_path, ".cursor/skills/x/SKILL.md", skill_md(name="x", description="the x skill"))

    discovery = hermetic(tmp_path)

    assert names(discovery.skills) == ["x"]
    skill = discovery.skills[0]
    assert skill.scope is SkillScope.CURSOR
    assert skill.source_path == ".cursor/skills/x/SKILL.md"
    assert skill.description == "the x skill"


def test_identity_is_the_folder_containing_skill_md(tmp_path):
    """A category folder is not a skill; the folder that HOLDS ``SKILL.md`` is (Cursor's rule)."""
    write(tmp_path, ".agents/skills/testing/api/SKILL.md", skill_md(description="api testing"))
    write(tmp_path, ".agents/skills/testing/notes.md", "just a note, not a SKILL.md")

    discovery = hermetic(tmp_path)

    # `testing` holds no SKILL.md, so it is a CATEGORY; `api` is the skill. Its name falls back to the
    # folder because the frontmatter declared none.
    assert names(discovery.skills) == ["api"]
    assert discovery.skills[0].source_path == ".agents/skills/testing/api/SKILL.md"


@pytest.mark.parametrize(
    ("directory", "scope"),
    [
        (".codna/skills", SkillScope.CODNA),
        (".cursor/skills", SkillScope.CURSOR),
        (".agents/skills", SkillScope.AGENTS),
        (".claude/skills", SkillScope.CLAUDE),
        (".codex/skills", SkillScope.CODEX),
    ],
)
def test_every_documented_directory_is_searched(tmp_path, directory, scope):
    """Cursor's directories, plus Claude/Codex compat, plus Codna's own — one reader, all of them."""
    write(tmp_path, f"{directory}/s/SKILL.md", skill_md(description="d"))

    discovery = hermetic(tmp_path)

    assert names(discovery.skills) == ["s"]
    assert discovery.skills[0].scope is scope


def test_search_order_is_codna_cursor_agents_then_compat():
    """The order is DATA, readable in one place, and it is also the precedence order."""
    entries = search_dirs("/repo", "/home", include_user=True)
    assert [entry.label for entry in entries] == [
        ".codna/skills", ".cursor/skills", ".agents/skills", ".claude/skills", ".codex/skills",
        "~/.codna/skills", "~/.cursor/skills", "~/.agents/skills", "~/.claude/skills",
        "~/.codex/skills",
    ]


def test_a_same_named_skill_is_shadowed_by_search_order_and_reported(tmp_path):
    """Two skills, one name: the first in search order wins, and the loser is NAMED."""
    write(tmp_path, ".cursor/skills/dup/SKILL.md", skill_md(name="dup", body="CURSOR"))
    write(tmp_path, ".codna/skills/nested/dup/SKILL.md", skill_md(name="dup", body="CODNA"))

    discovery = hermetic(tmp_path)

    assert len(discovery.skills) == 1
    assert discovery.skills[0].body == "CODNA"
    shadowed = [w for w in discovery.warnings if w.code == "skill_name_shadowed"]
    assert len(shadowed) == 1
    assert shadowed[0].source_path == ".cursor/skills/dup/SKILL.md"


def test_bundled_files_are_listed_relative_to_the_skill_folder(tmp_path):
    """A loaded skill has to name its material without depending on where the repo is checked out."""
    write(tmp_path, ".cursor/skills/pdf/SKILL.md", skill_md(description="pdf"))
    write(tmp_path, ".cursor/skills/pdf/scripts/fill.ts", "export const x = 1;\n")
    write(tmp_path, ".cursor/skills/pdf/references/schema.json", "{}\n")
    write(tmp_path, ".cursor/skills/pdf/unrelated.md", "not a bundled directory\n")

    skill = by_name(hermetic(tmp_path).skills, "pdf")

    # `scripts`, `references`, `assets` — in that order, the order Cursor documents. A file beside
    # them is not bundled material, so it is not listed.
    assert skill.bundled == ("scripts/fill.ts", "references/schema.json")


def test_a_missing_repository_is_reported_not_silently_empty(tmp_path):
    discovery = discover_skills(str(tmp_path / "nope"), include_user=False)
    assert discovery.skills == ()
    assert [warning.code for warning in discovery.warnings] == ["skills_root_missing"]


def test_a_symlinked_skill_file_outside_the_repository_is_skipped(tmp_path):
    """The security property, shared with the rule reader: a skill may not point outside the tree."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(skill_md(description="evil"), encoding="utf-8")
    skills = tmp_path / "repo" / ".cursor" / "skills"
    (skills / "evil").mkdir(parents=True)
    (skills / "evil" / "SKILL.md").symlink_to(outside / "SKILL.md")

    discovery = hermetic(tmp_path / "repo")

    assert discovery.skills == ()
    assert any(
        warning.code in ("skill_file_outside_repository", "skill_file_unreadable")
        for warning in discovery.warnings
    )


def test_a_bad_frontmatter_value_is_reported_and_the_skill_is_kept(tmp_path):
    """A wrong-typed switch must not silently become `false` (auto-attached) or `true` (manual)."""
    write(
        tmp_path,
        ".cursor/skills/x/SKILL.md",
        "---\ndescription: d\ndisable-model-invocation: maybe\n---\nBODY\n",
    )
    discovery = hermetic(tmp_path)

    assert names(discovery.skills) == ["x"]
    assert discovery.skills[0].disable_model_invocation is False
    assert any(w.code == "skill_frontmatter_invalid" for w in discovery.warnings)


def test_discovery_is_bounded_by_count(tmp_path, monkeypatch):
    """A tree with more than the cap reports the overflow rather than reading forever."""
    monkeypatch.setattr("codna.rules.skills.MAX_SKILLS", 3)
    for index in range(6):
        write(tmp_path, f".cursor/skills/s{index}/SKILL.md", skill_md(description=f"d{index}"))

    discovery = hermetic(tmp_path)

    assert len(discovery.skills) == 3
    overflow = [w for w in discovery.warnings if w.code == "skills_truncated_by_count"]
    assert len(overflow) == 1
    assert MAX_SKILLS == 500  # the shipped default, pinned so the monkeypatch cannot hide a change


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_paths_scoping
# ---------------------------------------------------------------------------------------------

def test_paths_scoping(tmp_path):
    """A skill with ``paths: "src/**/*.ts"`` is absent for a docs-only context."""
    write(
        tmp_path,
        ".cursor/skills/ts/SKILL.md",
        skill_md(name="ts", description="typescript", paths="src/**/*.ts"),
    )
    write(tmp_path, ".cursor/skills/always/SKILL.md", skill_md(name="always", description="any file"))
    discovery = hermetic(tmp_path)

    docs = skill_catalogue(discovery, files_in_context=["docs/a.md"])
    code = skill_catalogue(discovery, files_in_context=["src/app.ts"])

    assert names(docs.catalogued) == ["always"]
    assert "- ts" not in docs.text
    assert names(code.catalogued) == ["always", "ts"]
    assert "- ts: typescript — paths: src/**/*.ts" in code.text


def test_paths_scoping_reads_the_same_glob_language_as_a_rule(tmp_path):
    """The parity assertion: one matcher, so a skill's ``paths`` = a rule's ``globs``, exactly."""
    pattern = "src/**/*.{ts,tsx}"
    cases = [
        ("src/a.ts", True),
        ("src/deep/nested/a.tsx", True),
        ("src/a.css", False),
        ("docs/a.ts", False),
    ]
    write(
        tmp_path,
        ".cursor/skills/s/SKILL.md",
        skill_md(name="s", description="d", paths=pattern),
    )
    skill = by_name(hermetic(tmp_path).skills, "s")

    for path, expected in cases:
        rule_answer = globs_mod.matches_any((pattern,), path)
        skill_answer = skill.matches_any_file((path,))
        assert rule_answer is expected, f"the shared matcher changed for {path}"
        assert skill_answer is rule_answer, f"a skill disagreed with a rule about {path}"


def test_paths_scoping_is_unknown_not_empty_when_the_caller_says_nothing(tmp_path):
    """``None`` (the caller did not say) keeps a scoped skill; ``[]`` (nothing changed) drops it.

    This is the rule engine's own distinction, and a durable run depends on it: its agent may edit
    anything, so a ``paths``-scoped skill cannot be ruled out — while a review of an empty diff can.
    """
    write(
        tmp_path,
        ".cursor/skills/ts/SKILL.md",
        skill_md(name="ts", description="d", paths="src/**/*.ts"),
    )
    discovery = hermetic(tmp_path)

    unknown = skill_catalogue(discovery)
    empty = skill_catalogue(discovery, files_in_context=[])

    assert names(unknown.catalogued) == ["ts"]
    assert names(empty.catalogued) == []


def test_an_explicit_mention_catalogues_a_paths_scoped_skill_outside_its_scope(tmp_path):
    """``/skill-name`` is evidence enough: the user asked, so ``paths`` does not refuse the ask."""
    write(
        tmp_path,
        ".cursor/skills/ts/SKILL.md",
        skill_md(name="ts", description="d", paths="src/**/*.ts"),
    )
    discovery = hermetic(tmp_path)

    result = skill_catalogue(discovery, files_in_context=["docs/a.md"], explicit_mentions=["/ts"])

    assert names(result.catalogued) == ["ts"]


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_disable_model_invocation_is_manual_only
# ---------------------------------------------------------------------------------------------

def test_disable_model_invocation_is_manual_only(tmp_path):
    """Catalogued but not auto-attached: the entry says `/name`, and only the user can ask for it."""
    write(
        tmp_path,
        ".codna/skills/release/SKILL.md",
        skill_md(name="release", description="cut a release", manual=True),
    )
    write(
        tmp_path,
        ".codna/skills/lint/SKILL.md",
        skill_md(name="lint", description="lint the tree", manual=False),
    )
    discovery = hermetic(tmp_path)

    release = by_name(discovery.skills, "release")
    lint = by_name(discovery.skills, "lint")
    assert release.mode is SkillMode.MANUAL
    assert release.disable_model_invocation is True
    assert lint.mode is SkillMode.AUTO

    result = skill_catalogue(discovery)

    # BOTH are catalogued — a manual-only skill is invisible exactly when it needs to be invocable.
    assert names(result.catalogued) == ["lint", "release"]
    # …but the entry marks it, so nothing treats it as a candidate the model may pick unprompted.
    assert "- release [manual-only: /release]: cut a release" in result.text
    assert "[manual-only" not in result.text.split("- lint")[1].split("\n")[0]


def test_a_manual_only_skill_is_still_loadable_by_name(tmp_path):
    """The permission restricts who may ASK, not whether the answer exists."""
    write(
        tmp_path,
        ".codna/skills/release/SKILL.md",
        skill_md(name="release", description="d", manual=True, body="PUBLISH STEPS"),
    )
    skill = load_skill(str(tmp_path), "release", home=str(tmp_path / "no-home"), include_user=False)
    assert skill is not None
    assert skill.body == "PUBLISH STEPS"
    assert skill.mode is SkillMode.MANUAL


@pytest.mark.parametrize(
    "spelling",
    ["disable-model-invocation", "disable_model_invocation", "disableModelInvocation"],
)
def test_the_manual_switch_is_read_in_every_spelling_that_arrives(tmp_path, spelling):
    """Cursor documents kebab-case; hand-written YAML arrives snake and camel too."""
    write(
        tmp_path,
        ".cursor/skills/x/SKILL.md",
        f"---\nname: x\ndescription: d\n{spelling}: true\n---\nBODY\n",
    )
    skill = hermetic(tmp_path).skills[0]
    assert skill.disable_model_invocation is True


# ---------------------------------------------------------------------------------------------
# REQUIRED: test_catalogue_is_bounded
# ---------------------------------------------------------------------------------------------

def test_catalogue_is_bounded(tmp_path):
    """100 skills produce a catalogue under budget, with the BODIES not inlined."""
    for index in range(100):
        write(
            tmp_path,
            f".cursor/skills/s{index:03d}/SKILL.md",
            skill_md(
                name=f"s{index:03d}",
                description=f"skill number {index} that does something specific and useful",
                body=f"UNIQUE-BODY-MARKER-{index:03d}\n" + "x" * 2_000,
            ),
        )
    discovery = hermetic(tmp_path)
    assert len(discovery.skills) == 100

    budget = 4_096
    result = skill_catalogue(discovery, max_bytes=budget)

    # Under budget, measured in the same bytes the prompt is.
    assert result.used_bytes <= budget
    assert len(result.text.encode("utf-8")) <= budget
    # Some entries fit and the rest are NAMED — never silently missing.
    assert 0 < len(result.catalogued) < 100
    assert len(result.catalogued) + len(result.dropped) == 100
    # THE assertion: no skill body reached the catalogue, at any budget.
    assert "UNIQUE-BODY-MARKER" not in result.text
    assert "xxxxxxxxxx" not in result.text
    # …and the header names the pull, so the catalogue is actionable.
    assert CATALOGUE_HEADER.split("\n")[0] in result.text
    assert "codna_load_skill" in result.text


def test_the_full_bodies_cost_more_than_the_catalogue_by_orders_of_magnitude(tmp_path):
    """The number that makes progressive disclosure worth doing, measured on this fixture."""
    for index in range(60):
        write(
            tmp_path,
            f".cursor/skills/s{index:02d}/SKILL.md",
            skill_md(description=f"skill {index}", body="y" * 4_000),
        )
    discovery = hermetic(tmp_path)
    catalogue_bytes = len(skill_catalogue(discovery, max_bytes=1 << 20).text.encode("utf-8"))
    bodies_bytes = sum(len(skill.body.encode("utf-8")) for skill in discovery.skills)
    assert catalogue_bytes < bodies_bytes / 50


def test_the_catalogue_does_not_starve_a_small_entry_behind_a_big_one():
    """One enormous description must not hide every entry after it."""
    lines = [
        CatalogueLine(name="a-big", source_path="a", line="- a-big: " + "y" * 3_000),
        CatalogueLine(name="b-small", source_path="b", line="- b-small: small"),
    ]
    result = render_catalogue(lines, header="HEADER", max_bytes=200)
    assert [entry.name for entry in result.kept] == ["b-small"]
    assert [entry.name for entry in result.dropped] == ["a-big"]


def test_a_zero_budget_reports_everything_rather_than_emitting_an_unbudgeted_list():
    lines = [CatalogueLine(name="a", source_path="a", line="- a: d")]
    result = render_catalogue(lines, header="HEADER", max_bytes=0)
    assert result.text == ""
    assert result.kept == ()
    assert [entry.name for entry in result.dropped] == ["a"]


def test_the_catalogue_is_deterministic(tmp_path):
    for index in range(20):
        write(tmp_path, f".cursor/skills/s{index}/SKILL.md", skill_md(description=f"d{index}"))
    discovery = hermetic(tmp_path)

    first = skill_catalogue(discovery, max_bytes=300)
    second = skill_catalogue(discovery, max_bytes=300)

    assert first.text == second.text
    assert names(first.catalogued) == names(second.catalogued)
    assert names(first.dropped) == names(second.dropped)


def test_include_catalog_false_carries_nothing(tmp_path):
    write(tmp_path, ".cursor/skills/x/SKILL.md", skill_md(name="x", description="d"))
    result = skill_catalogue(hermetic(tmp_path), include_catalog=False)
    assert result.text == ""
    assert result.catalogued == ()


# ---------------------------------------------------------------------------------------------
# the pull, and one budget with the rules
# ---------------------------------------------------------------------------------------------

def test_load_skill_pulls_by_the_name_the_catalogue_printed(tmp_path):
    write(
        tmp_path,
        ".cursor/skills/pdf/SKILL.md",
        skill_md(name="pdf", description="d", body="# PDF\nUSE THE SCRIPT"),
    )
    skill = load_skill(str(tmp_path), "pdf", home=str(tmp_path / "no-home"), include_user=False)
    assert skill is not None
    assert skill.body.startswith("# PDF")
    assert "USE THE SCRIPT" in skill.body


@pytest.mark.parametrize("written", ["pdf", "/pdf", "@pdf", "PDF", "pdf.md"])
def test_load_skill_accepts_the_shapes_a_caller_types(tmp_path, written):
    write(tmp_path, ".cursor/skills/pdf/SKILL.md", skill_md(name="pdf", description="d"))
    assert load_skill(
        str(tmp_path), written, home=str(tmp_path / "no-home"), include_user=False,
    ) is not None


def test_load_skill_returns_none_for_an_unknown_name(tmp_path):
    """``None`` is a real answer, and the caller names it in its own error."""
    assert load_skill(
        str(tmp_path), "nope", home=str(tmp_path / "no-home"), include_user=False,
    ) is None


def test_load_context_puts_rules_and_the_catalogue_in_one_text_under_one_budget(tmp_path):
    """The join that makes this a feature: both halves, one ceiling, one account of what did not fit."""
    write(tmp_path, ".codna/rules/python.mdc", "---\nalwaysApply: true\n---\nPYTHON RULE\n")
    write(tmp_path, ".cursor/skills/pdf/SKILL.md", skill_md(name="pdf", description="fill a form"))

    loaded = load_context(
        str(tmp_path), files_in_context=["src/app.ts"], max_bytes=4_096,
        home=str(tmp_path / "no-home"), include_user=False,
    )

    assert "PYTHON RULE" in loaded.text
    assert "- pdf: fill a form" in loaded.text
    assert loaded.text.index("PYTHON RULE") < loaded.text.index("- pdf")
    assert loaded.used_bytes <= loaded.budget_bytes == 4_096
    assert loaded.skill_catalogue.budget_bytes == 4_096 - loaded.rules.result.used_bytes - 2


def test_the_rules_wins_the_budget_and_the_catalogue_degrades(tmp_path):
    """Instructions survive, the index degrades — never the reverse."""
    write(tmp_path, ".codna/rules/python.mdc", "---\nalwaysApply: true\n---\n" + "R" * 800 + "\n")
    for index in range(40):
        write(
            tmp_path,
            f".cursor/skills/s{index:02d}/SKILL.md",
            skill_md(description=f"skill number {index} with a description of realistic length"),
        )
    loaded = load_context(
        str(tmp_path), max_bytes=2_000, home=str(tmp_path / "no-home"), include_user=False,
    )
    assert "R" * 200 in loaded.text  # the rule body is intact
    assert loaded.skill_catalogue.used_bytes <= 2_000 - 800
    assert loaded.skill_catalogue.dropped  # and the overflow is named
    assert len(loaded.skill_catalogue.catalogued) < 40


def test_a_repository_with_no_skills_costs_nothing(tmp_path):
    """The ordinary case for most repositories: no skills, no text, no subprocess."""
    write(tmp_path, ".codna/rules/python.mdc", "---\nalwaysApply: true\n---\nPYTHON\n")
    loaded = load_context(
        str(tmp_path), max_bytes=4_096, home=str(tmp_path / "no-home"), include_user=False,
    )
    assert loaded.text.strip() == "# .codna/rules/python.mdc\nPYTHON"
    assert loaded.skills.skills == ()
    assert loaded.skill_catalogue.text == ""


def test_include_skills_false_is_the_pre_skill_answer(tmp_path):
    write(tmp_path, ".cursor/skills/pdf/SKILL.md", skill_md(name="pdf", description="d"))
    loaded = load_context(
        str(tmp_path), max_bytes=4_096, home=str(tmp_path / "no-home"), include_user=False,
        include_skills=False,
    )
    assert loaded.skill_catalogue.text == ""
    assert "pdf" not in loaded.text


# ---------------------------------------------------------------------------------------------
# the fixture repository — the demonstration, pinned so it cannot rot
# ---------------------------------------------------------------------------------------------

def test_the_fixture_repo_discovers_all_three_skills():
    discovery = discover_skills(str(FIXTURE), home=str(FIXTURE / "no-home"), include_user=False)
    assert names(discovery.skills) == ["release", "pdf", "api"]  # codna → cursor → agents


def test_the_fixture_scopes_on_the_files_in_context():
    discovery = discover_skills(str(FIXTURE), home=str(FIXTURE / "no-home"), include_user=False)

    docs = skill_catalogue(discovery, files_in_context=["docs/readme.md"], max_bytes=4_096)
    code = skill_catalogue(discovery, files_in_context=["src/app.ts"], max_bytes=4_096)

    assert names(docs.catalogued) == ["release", "api"]  # `pdf` is scoped to src/**/*.ts
    assert names(code.catalogued) == ["release", "pdf", "api"]


def test_the_fixture_release_skill_is_manual_only_and_pdf_ships_a_script():
    discovery = discover_skills(str(FIXTURE), home=str(FIXTURE / "no-home"), include_user=False)
    assert by_name(discovery.skills, "release").mode is SkillMode.MANUAL
    assert by_name(discovery.skills, "pdf").bundled == ("scripts/fill.ts",)
    assert by_name(discovery.skills, "api").source_path == ".agents/skills/testing/api/SKILL.md"
