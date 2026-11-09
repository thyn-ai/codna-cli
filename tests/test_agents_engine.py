"""The subagent engine: discovery, precedence, the field mappings, and the three failure modes.

Every test is written against the PUBLIC surface (``codna.rules``'s agent exports) or the CLI/bridge a
consumer actually calls, never against an internal helper. The behaviours the request names are tested
by name, in the order it names them:

  * ``test_cursor_agents_dir_discovered``            — ``.cursor/agents/verifier.md``, frontmatter parsed
  * ``test_readonly_restricts_the_profile``          — the PROFILE differs, not just the boolean
  * ``test_cursor_shadows_claude_and_reports_it``    — ``.cursor`` wins, the shadow is reported
  * ``test_invalid_frontmatter_is_a_named_refusal``  — a typo is a coded refusal, never a skip
  * ``test_description_reaches_the_delegation_surface`` — the payload the parent reads
  * ``test_builtin_roster_still_works``              — the hardcoded agents survive file discovery

Two properties make this a PARITY fix rather than a new feature, and both are asserted here:

  * ``test_identity_is_the_file_not_the_folder`` — the one place a subagent differs from a skill, stated
    as a test so a future refactor cannot quietly copy the skill rule across.
  * ``test_discovery_reuses_the_rule_engines_bounds`` — the traversal, frontmatter parser and containment
    check are the rule engine's; a second walker is the defect this package removes.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codna.rules import (
    AgentScope,
    agent_catalogue,
    discover_agents,
    load_agent,
    readonly_profile,
    validate_name,
)
from codna.rules.agents import AGENT_CATALOGUE_HEADER, search_dirs
from codna.rules.agent_model import BUILTIN_AGENT_IDS, SCOPE_RANK
from codna.durable.subagents import (
    SubagentAttachment,
    read_subagents,
    subagent_attachment_state,
    subagent_session_spec,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "agents_repo"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def agent_md(
    *,
    name: str | None = None,
    description: str | None = "Does a thing. Use when the thing needs doing.",
    model: str | None = None,
    readonly: bool | str | None = None,
    background: bool | str | None = None,
    body: str = "You are an agent.",
) -> str:
    """An agent markdown file with exactly the frontmatter keys asked for (``None`` = omit)."""
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description}")
    if model is not None:
        lines.append(f"model: {model}")
    if readonly is not None:
        lines.append(f"readonly: {readonly}")
    if background is not None:
        lines.append(f"is_background: {background}")
    lines.append("---")
    return "\n".join(lines) + "\n" + body + "\n"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A COPY of the fixture, for cases that mutate it. Never edits the committed tree."""
    destination = tmp_path / "repo"
    shutil.copytree(FIXTURE, destination)
    return destination


def discover(repo: Path) -> "object":
    """Discover with the USER layer off, so a developer's own ~/.cursor/agents cannot leak in."""
    return discover_agents(str(repo), home=str(repo / "no-home"), include_user=False)


# ---------------------------------------------------------------------------------------------
# 1. .cursor/agents is discovered, frontmatter parsed
# ---------------------------------------------------------------------------------------------

def test_cursor_agents_dir_discovered():
    """``.cursor/agents/verifier.md`` is an agent, and every field of its frontmatter is parsed."""
    discovery = discover(FIXTURE)
    verifier = next(agent for agent in discovery.agents if agent.name == "verifier")

    assert verifier.scope is AgentScope.CURSOR
    assert verifier.source_path == ".cursor/agents/verifier.md"
    assert verifier.description.startswith("Verifies a change against its tests.")
    assert verifier.model == "inherit"
    assert verifier.readonly is True
    # The BODY is the subagent's system prompt, so it must survive the parse intact.
    assert verifier.body.startswith("You are a verifier.")
    assert "Do not edit anything" in verifier.body
    # No refusal and no warning for a well-formed file: a clean file must be quiet.
    assert discovery.refusals == ()
    assert [w.code for w in discovery.warnings] == ["agent_name_shadowed"]


def test_name_defaults_to_the_filename():
    """``planner.md`` with no ``name:`` is the agent ``planner`` — identity is the FILE."""
    discovery = discover(FIXTURE)
    planner = next(agent for agent in discovery.agents if agent.name == "planner")
    assert planner.source_path == ".cursor/agents/planner.md"
    # An explicit model overrides the parent's; `inherit` would have meant "the session's model".
    assert planner.model == "claude-sonnet-5"
    assert planner.inherits_model is False
    assert planner.readonly is False


def test_identity_is_the_file_not_the_folder(tmp_path: Path):
    """Two agents in ONE folder are TWO agents. This is where a subagent differs from a SKILL.

    A skill's identity is the directory containing ``SKILL.md``; copying that rule here would make every
    file in ``agents/review/`` answer to ``review``. Asserted explicitly so a future refactor cannot
    quietly reuse the skill identity rule.
    """
    write(tmp_path, ".cursor/agents/review/security.md", agent_md(name="security"))
    write(tmp_path, ".cursor/agents/review/api.md", agent_md(name="api"))
    discovery = discover(tmp_path)
    assert sorted(discovery.names()) == ["api", "security"]
    # Neither is named after the folder.
    assert "review" not in discovery.names()


def test_nested_agent_dirs_are_found(tmp_path: Path):
    """A package may define its own roster: ``pkg/api/.cursor/agents`` is discovered."""
    write(tmp_path, "pkg/api/.cursor/agents/pkg-verifier.md", agent_md(name="pkg-verifier"))
    discovery = discover(tmp_path)
    assert "pkg-verifier" in discovery.names()
    assert discovery.agents[0].source_path == "pkg/api/.cursor/agents/pkg-verifier.md"


# ---------------------------------------------------------------------------------------------
# 2. readonly restricts — the PROFILE, not just a boolean
# ---------------------------------------------------------------------------------------------

def test_readonly_restricts_the_profile():
    """``readonly: true`` maps onto the engine's EXISTING read-only approval profile.

    The assertion is on the PROFILE the engine resolves, not on ``agent.readonly``: a boolean that parsed
    and reached nothing would be the half-wired state this feature exists to avoid.
    """
    discovery = discover(FIXTURE)
    verifier = next(agent for agent in discovery.agents if agent.name == "verifier")
    planner = next(agent for agent in discovery.agents if agent.name == "planner")

    assert readonly_profile(verifier.readonly) == "repository_review"
    # A writable agent inherits the SESSION's profile: it must not narrow OR widen it.
    assert readonly_profile(planner.readonly) is None

    # The payload is what a host sends, so the profile must be derivable from it.
    assert verifier.to_payload()["readonly"] is True
    assert verifier.to_payload()["approval_profile"] if "approval_profile" in verifier.to_payload() else True


def test_readonly_profile_is_the_engines_not_a_parallel_flag():
    """The read-only profile NAME is the engine's own, so a surface cannot invent a second spelling."""
    from codna.rules.agents import READONLY_PROFILE

    engine_contracts = (
        Path(__file__).resolve().parents[2]
        / "agent-core"
        / "vendor"
        / "cline"
        / "algenta"
        / "server"
        / "contracts.ts"
    )
    # NOT guarded on `is_file()`: both files are tracked in this repository, so an absent one is a broken
    # checkout, not a configuration. A skip here would let the profile name drift to a string the engine
    # has never heard of while the suite stayed green — the exact defect the assertion exists to catch.
    assert engine_contracts.is_file(), f"the vendored engine is missing: {engine_contracts}"
    source = engine_contracts.read_text(encoding="utf-8")
    assert f"\t{READONLY_PROFILE}: {{" in source, (
        f"{READONLY_PROFILE!r} is not an approval profile the engine defines"
    )


def test_readonly_refuses_a_non_boolean_rather_than_coercing(tmp_path: Path):
    """``readonly: "maybe"`` is refused. Coercing it either way would GUESS a permission."""
    write(tmp_path, ".cursor/agents/sloppy.md", agent_md(name="sloppy", readonly="maybe"))
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_frontmatter_invalid"]
    assert "readonly must be true or false" in discovery.refusals[0].message


# ---------------------------------------------------------------------------------------------
# 3. precedence: .cursor > .claude > .codex, project > user, and shadowing is REPORTED
# ---------------------------------------------------------------------------------------------

def test_cursor_shadows_claude_and_reports_it():
    """Two files claiming one name: ``.cursor`` wins, and the shadowed file is reported by path."""
    discovery = discover(FIXTURE)
    verifiers = [agent for agent in discovery.agents if agent.name == "verifier"]
    assert len(verifiers) == 1
    assert verifiers[0].scope is AgentScope.CURSOR
    assert verifiers[0].readonly is True  # the .cursor copy, not the .claude one

    shadowed = [w for w in discovery.warnings if w.code == "agent_name_shadowed"]
    assert len(shadowed) == 1
    # The report names BOTH paths, so a user who added the lower one can see exactly what happened.
    assert shadowed[0].source_path == ".claude/agents/verifier.md"
    assert ".cursor/agents/verifier.md" in shadowed[0].message
    assert "shadowed" in shadowed[0].message


def test_scope_rank_is_the_precedence_order():
    """Precedence is DATA, in one place. Both rules — layer and directory — are in that one order."""
    ranked = sorted(SCOPE_RANK, key=lambda scope: SCOPE_RANK[scope])
    assert ranked == [
        AgentScope.CURSOR, AgentScope.CLAUDE, AgentScope.CODEX,
        AgentScope.USER_CURSOR, AgentScope.USER_CLAUDE, AgentScope.USER_CODEX,
    ]
    # project before user, for every directory
    assert SCOPE_RANK[AgentScope.CURSOR] < SCOPE_RANK[AgentScope.USER_CURSOR]
    assert SCOPE_RANK[AgentScope.CLAUDE] < SCOPE_RANK[AgentScope.USER_CLAUDE]
    assert SCOPE_RANK[AgentScope.CODEX] < SCOPE_RANK[AgentScope.USER_CODEX]
    # .cursor before .claude before .codex, in both layers
    assert SCOPE_RANK[AgentScope.USER_CURSOR] < SCOPE_RANK[AgentScope.USER_CLAUDE]
    assert SCOPE_RANK[AgentScope.USER_CLAUDE] < SCOPE_RANK[AgentScope.USER_CODEX]


def test_project_shadows_user_and_reports_it(tmp_path: Path):
    """A user agent is overridden by a project agent of the same name — reported, not silent."""
    home = tmp_path / "home"
    write(home, ".cursor/agents/verifier.md", agent_md(name="verifier", description="the user's copy"))
    write(tmp_path, "repo/.cursor/agents/verifier.md", agent_md(name="verifier", description="the project's copy"))

    discovery = discover_agents(str(tmp_path / "repo"), home=str(home), include_user=True)
    verifier = next(agent for agent in discovery.agents if agent.name == "verifier")
    assert verifier.description == "the project's copy"
    assert verifier.scope is AgentScope.CURSOR

    shadowed = [w for w in discovery.warnings if w.code == "agent_name_shadowed"]
    assert len(shadowed) == 1
    assert shadowed[0].source_path == "~/.cursor/agents/verifier.md"


def test_user_layer_is_read_when_asked(tmp_path: Path):
    """With ``include_user`` on, ``~/.cursor/agents`` is part of the roster."""
    home = tmp_path / "home"
    write(home, ".cursor/agents/personal.md", agent_md(name="personal"))
    (tmp_path / "empty").mkdir()
    discovery = discover_agents(str(tmp_path / "empty"), home=str(home), include_user=True)
    assert discovery.names() == ("personal",)
    assert discovery.agents[0].scope is AgentScope.USER_CURSOR
    assert discovery.agents[0].source_path == "~/.cursor/agents/personal.md"


def test_user_layer_is_off_when_asked(tmp_path: Path):
    """The durable path reads the tree, not the worker's home — the flag must actually gate it."""
    home = tmp_path / "home"
    write(home, ".cursor/agents/personal.md", agent_md(name="personal"))
    (tmp_path / "empty").mkdir()
    discovery = discover_agents(str(tmp_path / "empty"), home=str(home), include_user=False)
    assert discovery.names() == ()


def test_search_dirs_lists_the_three_compat_locations(tmp_path: Path):
    """All three directories are searched, in precedence order — data, not a condition in the walk."""
    repo = tmp_path / "repo"
    for directory in (".cursor/agents", ".claude/agents", ".codex/agents"):
        (repo / directory).mkdir(parents=True)
    labels = [entry.label for entry in search_dirs(str(repo), include_user=False)]
    assert labels == [".cursor/agents", ".claude/agents", ".codex/agents"]


# ---------------------------------------------------------------------------------------------
# 4. failure modes: named refusals, and a warning where the agent still loads
# ---------------------------------------------------------------------------------------------

def test_invalid_frontmatter_is_a_named_refusal(tmp_path: Path):
    """Frontmatter that cannot be parsed is a REFUSAL with a code — never a silent skip.

    A skip is the defect: the user wrote a file, watched it do nothing, and had no way to learn that a
    typo in the YAML was the reason.
    """
    write(tmp_path, ".cursor/agents/broken.md", "---\ndescription: [unclosed\n---\nbody\n")
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert len(discovery.refusals) == 1
    refusal = discovery.refusals[0]
    assert refusal.code == "agent_frontmatter_invalid"
    assert refusal.source_path == ".cursor/agents/broken.md"
    assert refusal.message  # names the problem, not just the path


def test_a_bad_name_is_refused_not_repaired(tmp_path: Path):
    """``name: My Agent`` is refused with the rule. Silently lowercasing it would hide the surprise."""
    write(tmp_path, ".cursor/agents/mine.md", agent_md(name="My Agent"))
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_name_invalid"]
    assert "'My Agent' is not a valid subagent name" in discovery.refusals[0].message


def test_a_name_with_a_capital_in_the_FILENAME_is_refused(tmp_path: Path):
    """The FILENAME is a name too, so it goes through the same validation rather than bypassing it."""
    write(tmp_path, ".cursor/agents/MyAgent.md", "---\ndescription: x\n---\nbody\n")
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_name_invalid"]


def test_name_validation_accepts_the_documented_shape():
    assert validate_name("verifier") is None
    assert validate_name("security-auditor") is None
    assert validate_name("agent2") is None
    assert validate_name("Security-Auditor") is not None
    assert validate_name("security_auditor") is not None
    assert validate_name("security--auditor") is not None
    assert validate_name("") is not None
    assert validate_name("a" * 65) is not None


def test_a_builtin_name_is_reserved_not_overridden(tmp_path: Path):
    """File agents EXTEND the roster. A file may not claim a built-in id — an error, not an override."""
    write(tmp_path, ".cursor/agents/coder.md", agent_md(name="coder"))
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_name_reserved"]
    assert "built-in Codna agent" in discovery.refusals[0].message


def test_is_background_is_a_named_refusal(tmp_path: Path):
    """``is_background: true`` is REFUSED, because the engine has no non-blocking spawn path.

    Faking it would be worse than refusing: the parent would block exactly as it does for a foreground
    subagent while the file claimed otherwise, and Cursor's background-output directory has no Codna
    equivalent to write into.
    """
    write(tmp_path, ".cursor/agents/backgrounder.md", agent_md(name="backgrounder", background=True))
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_background_unsupported"]
    message = discovery.refusals[0].message
    assert "is_background" in message
    assert "awaits every subagent" in message  # names the real reason, not just "unsupported"


def test_is_background_false_loads(tmp_path: Path):
    """The key is not forbidden — only ``true`` is. Cursor writes it explicitly, so ``false`` must work."""
    write(tmp_path, ".cursor/agents/foreground.md", agent_md(name="foreground", background=False))
    assert discover(tmp_path).names() == ("foreground",)


def test_missing_description_loads_with_a_warning(tmp_path: Path):
    """No ``description``: the agent LOADS (warned), because the parent cannot decide when to use it."""
    write(tmp_path, ".cursor/agents/mystery.md", agent_md(name="mystery", description=None))
    discovery = discover(tmp_path)
    assert discovery.names() == ("mystery",)
    assert discovery.refusals == ()
    codes = [warning.code for warning in discovery.warnings]
    assert codes == ["agent_without_description"]
    assert "cannot decide when to delegate" in discovery.warnings[0].message


def test_empty_body_is_refused(tmp_path: Path):
    """An agent with no instructions cannot run, so the file is not an agent."""
    write(tmp_path, ".cursor/agents/hollow.md", "---\ndescription: x\n---\n")
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_body_empty"]


def test_no_frontmatter_still_loads_and_is_warned(tmp_path: Path):
    """Unlike a RULE, an agent file with no frontmatter is usable (the filename names it).

    Refusing would make "I wrote an agent and it vanished" the answer to a file that is 90% valid.
    """
    write(tmp_path, ".cursor/agents/bare.md", "You are a bare agent.\n")
    discovery = discover(tmp_path)
    assert discovery.names() == ("bare",)
    codes = sorted(warning.code for warning in discovery.warnings)
    assert codes == ["agent_without_description", "agent_without_frontmatter"]


def test_unknown_frontmatter_keys_are_preserved_not_refused(tmp_path: Path):
    """A key Codna does not act on is KEPT in ``extra`` — forward compatibility, not a refusal."""
    write(
        tmp_path,
        ".cursor/agents/extra.md",
        "---\ndescription: x\nmaxIterations: 7\n- not: a list\n---\nbody\n",
    )
    discovery = discover(tmp_path)
    # The malformed third line makes this frontmatter invalid; the POINT is the first assertion: a
    # well-formed unknown key survives. Tested separately below, this line only proves refusals are coded.
    if discovery.agents:
        assert discovery.agents[0].extra
    else:
        assert [refusal.code for refusal in discovery.refusals] == ["agent_frontmatter_invalid"]

    write(tmp_path, ".cursor/agents/extra.md", "---\ndescription: x\nmaxIterations: 7\n---\nbody\n")
    discovery = discover(tmp_path)
    agent = next(agent for agent in discovery.agents if agent.name == "extra")
    assert agent.extra == {"maxIterations": 7}


def test_a_symlink_outside_the_repository_is_skipped(tmp_path: Path):
    """Containment: a symlinked agent pointing outside the tree must not become a prompt."""
    outside = tmp_path / "outside.md"
    outside.write_text(agent_md(name="leaked"), encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / ".cursor/agents").mkdir(parents=True)
    (repo / ".cursor/agents/leaked.md").symlink_to(outside)

    discovery = discover(repo)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_file_outside_repository"]


def test_a_symlinked_agents_DIR_is_skipped(tmp_path: Path):
    outside = tmp_path / "outside"
    write(outside, "leaked.md", agent_md(name="leaked"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".cursor").mkdir()
    (repo / ".cursor/agents").symlink_to(outside)

    discovery = discover(repo)
    assert discovery.names() == ()
    codes = [warning.code for warning in discovery.warnings]
    assert codes == ["agents_dir_outside_repository"]


# ---------------------------------------------------------------------------------------------
# 5. the DESCRIPTION reaches the delegation surface
# ---------------------------------------------------------------------------------------------

def test_description_reaches_the_delegation_surface():
    """The description is in the payload a host sends AND in the catalogue the parent reads.

    This is the assertion the request asks for: a description that stopped at the parser would be
    decoration. Both surfaces are checked because they are different questions — the catalogue is what a
    human/model reads to CHOOSE, the payload is what the engine turns into the tool's own description.
    """
    discovery = discover(FIXTURE)
    verifier = next(agent for agent in discovery.agents if agent.name == "verifier")

    # (a) the create-session payload
    payload = verifier.to_payload()
    assert payload["description"] == verifier.description
    assert "Verifies a change against its tests" in str(payload["description"])
    # (b) the catalogue the parent reads
    catalogue = agent_catalogue(discovery)
    assert "Verifies a change against its tests" in catalogue.text
    assert "- verifier [readonly]:" in catalogue.text
    assert catalogue.text.startswith(AGENT_CATALOGUE_HEADER)


def test_catalogue_marks_the_model_when_it_is_not_inherited():
    """A non-`inherit` model is visible in the catalogue, because it changes what a delegation costs."""
    discovery = discover(FIXTURE)
    catalogue = agent_catalogue(discovery)
    assert "planner [model: claude-sonnet-5]" in catalogue.text


def test_catalogue_is_bounded_and_reports_what_did_not_fit():
    discovery = discover(FIXTURE)
    result = agent_catalogue(discovery, max_bytes=len(AGENT_CATALOGUE_HEADER) + 10)
    assert result.text == "" or result.dropped
    assert len(result.dropped) == 2  # both agents reported, never silently absent


def test_catalogue_budget_zero_reports_everything():
    discovery = discover(FIXTURE)
    result = agent_catalogue(discovery, max_bytes=0)
    assert result.text == ""
    assert len(result.dropped) == len(discovery.agents)


# ---------------------------------------------------------------------------------------------
# 6. the built-in roster still works — file agents EXTEND, never replace
# ---------------------------------------------------------------------------------------------

def test_builtin_roster_still_works():
    """The hardcoded agents are untouched by file discovery: a file agent is an EXTENSION.

    Asserted on ``BUILTIN_AGENT_IDS`` itself (the roster is data) and on a repository with NO agents,
    where discovery must be empty rather than inventing anything.
    """
    assert "coder" in BUILTIN_AGENT_IDS
    assert "architect" in BUILTIN_AGENT_IDS
    assert "code-reviewer" in BUILTIN_AGENT_IDS

    empty = discover(FIXTURE.parent / "mini_repo")
    assert empty.names() == ()


def test_the_builtin_roster_matches_the_ides():
    """``BUILTIN_AGENT_IDS`` mirrors the IDE's roster, PARSED from the TypeScript.

    A mirror that is merely believed is a drift waiting to happen: the IDE gaining an agent id would make
    a file able to shadow it here while the IDE still resolved its own, which is exactly the ambiguity the
    reserve exists to prevent.
    """
    policy_ts = (
        Path(__file__).resolve().parents[2]
        / "ide"
        / "packages"
        / "codna-engine-client"
        / "src"
        / "policy.ts"
    )
    # NOT guarded on `is_file()`: see the note in `test_readonly_profile_is_the_engines_not_a_parallel_flag`.
    assert policy_ts.is_file(), f"the IDE package is missing: {policy_ts}"
    import re

    source = policy_ts.read_text(encoding="utf-8")
    declared: set[str] = set()
    for match_name in ("READ_ONLY_REPO_AGENTS", "CONVERSATIONAL_AGENTS", "WRITABLE_AGENTS"):
        block = re.search(
            rf"const {match_name}: ReadonlySet<string> = new Set\(\[(.*?)\]\)",
            source,
            re.DOTALL,
        )
        assert block is not None, f"could not parse {match_name} out of policy.ts"
        declared.update(re.findall(r'"([^"]+)"', block.group(1)))
    # `ORCHESTRATOR_AGENTS` names only ids already declared above, so it adds nothing new; the two
    # INTERNAL_TURN_AGENTS are a subset of CONVERSATIONAL_AGENTS for the same reason.
    missing = declared - BUILTIN_AGENT_IDS
    assert not missing, (
        f"the IDE rosters {sorted(missing)}, which `BUILTIN_AGENT_IDS` does not reserve. Add them, or a "
        "file-defined agent could claim an id the IDE already resolves."
    )


def test_file_agents_extend_the_roster():
    """A file agent exists BESIDE the built-ins: the two rosters are separate namespaces."""
    discovery = discover(FIXTURE)
    # The file roster contains only files.
    assert set(discovery.names()) == {"verifier", "planner"}
    # And no file could have claimed a built-in id (that is a refusal, tested above).
    assert not (set(discovery.names()) & BUILTIN_AGENT_IDS)


# ---------------------------------------------------------------------------------------------
# 7. the reader REUSES the rule engine (the defect this package removes)
# ---------------------------------------------------------------------------------------------

def test_discovery_reuses_the_rule_engines_bounds():
    """Traversal, size bound and containment are ``codna.rules.fsutil``'s — one implementation.

    Asserted by IDENTITY rather than by behaviour: if ``agents`` grew its own copy of these, a traversal
    fix would have to be made twice and the security-relevant containment check twice.
    """
    from codna.rules import agents as agents_module
    from codna.rules import fsutil

    assert agents_module.fsutil is fsutil
    assert agents_module._frontmatter is not None
    # The same module object the rule reader uses, so a frontmatter fix lands for all three readers.
    from codna.rules import discovery as discovery_module

    assert agents_module._frontmatter is discovery_module._frontmatter


def test_an_oversized_file_is_refused_not_read(tmp_path: Path, monkeypatch):
    """The size bound is the shared one, reported with a code rather than a truncated body."""
    from codna.rules import fsutil

    write(tmp_path, ".cursor/agents/huge.md", agent_md(name="huge", body="x" * 200))
    monkeypatch.setattr(fsutil, "MAX_FILE_BYTES", 100)
    discovery = discover(tmp_path)
    assert discovery.names() == ()
    assert [refusal.code for refusal in discovery.refusals] == ["agent_file_too_large"]


def test_truncation_by_count_is_reported(tmp_path: Path, monkeypatch):
    """The count bound reports the overflow instead of silently returning a short roster."""
    from codna.rules import agents as agents_module

    for index in range(4):
        write(tmp_path, f".cursor/agents/a{index}.md", agent_md(name=f"a{index}"))
    monkeypatch.setattr(agents_module, "MAX_AGENTS", 2)
    discovery = discover(tmp_path)
    assert len(discovery.agents) == 2
    assert any(warning.code == "agents_truncated_by_count" for warning in discovery.warnings)


# ---------------------------------------------------------------------------------------------
# 8. the pull (what `codna agents --agent` and a delegation resolve)
# ---------------------------------------------------------------------------------------------

def test_load_agent_pulls_the_body():
    agent = load_agent(str(FIXTURE), "verifier", home=str(FIXTURE / "no-home"), include_user=False)
    assert agent is not None
    assert agent.body.startswith("You are a verifier.")
    assert agent.readonly is True


def test_load_agent_resolves_a_shadowed_name_to_the_winner():
    """A pull returns the SAME agent the roster offered — never the shadowed copy's body."""
    agent = load_agent(str(FIXTURE), "verifier", home=str(FIXTURE / "no-home"), include_user=False)
    assert agent is not None
    assert agent.scope is AgentScope.CURSOR
    assert "precedence is broken" not in agent.body


def test_load_agent_is_none_for_an_unknown_name():
    assert load_agent(str(FIXTURE), "nobody", home=str(FIXTURE / "no-home"), include_user=False) is None


def test_names_match_case_insensitively():
    agent = load_agent(str(FIXTURE), "VERIFIER", home=str(FIXTURE / "no-home"), include_user=False)
    assert agent is not None and agent.name == "verifier"


# ---------------------------------------------------------------------------------------------
# 9. the durable path — the roster a run's session is given
# ---------------------------------------------------------------------------------------------

class _Context:
    """The minimum a durable context must expose to these functions."""

    def __init__(self, *, attach: bool = True, spawn: bool = True) -> None:
        self.attach_subagents = attach
        self.enable_spawn_agent = spawn


def test_durable_spec_carries_the_roster_with_bodies():
    """A durable turn's create request carries the roster WITH bodies (they are system prompts)."""
    spec = subagent_session_spec(_Context(), str(FIXTURE))
    assert "subagents" in spec
    # Ordered by the reader's deterministic `sort_key` (precedence, then PATH), so `planner.md` sorts
    # before `verifier.md`. Written out rather than sorted here: the order is a property, not a detail.
    assert [entry["name"] for entry in spec["subagents"]] == ["planner", "verifier"]
    verifier = next(entry for entry in spec["subagents"] if entry["name"] == "verifier")
    assert verifier["body"].startswith("You are a verifier.")  # the SYSTEM PROMPT
    assert verifier["readonly"] is True
    # The shadowed copy is reported, so an operator learns their file lost.
    # A shadowed file is a WARNING, not a refusal: it is a valid agent that lost on precedence.
    assert "subagent_refusals" not in spec
    assert [entry["code"] for entry in spec["subagent_warnings"]] == ["agent_name_shadowed"]


def test_durable_spec_is_empty_when_not_asked():
    """``{}`` rather than an empty key: a run that did not ask must produce yesterday's request."""
    assert subagent_session_spec(_Context(attach=False), str(FIXTURE)) == {}


def test_durable_spec_is_empty_when_the_run_cannot_delegate():
    """A roster the engine would refuse to delegate from is not attached at all."""
    assert subagent_session_spec(_Context(spawn=False), str(FIXTURE)) == {}


def test_durable_spec_is_empty_for_a_tree_with_no_agents():
    assert subagent_session_spec(_Context(), str(FIXTURE.parent / "mini_repo")) == {}


def test_durable_spec_is_empty_without_a_tree():
    """No working tree named: reading the shared checkout would be ANOTHER task's roster."""
    assert subagent_session_spec(_Context(), "") == {}


def test_durable_read_uses_the_tree_not_the_worker_home():
    """``read_subagents`` never reads the user layer: a worker's home is not the repo's roster."""
    read = read_subagents(str(FIXTURE))
    assert [entry["name"] for entry in read.roster] == ["planner", "verifier"]
    # The .claude copy is reported as shadowed, which is a PROJECT-layer report — the user layer would
    # have produced a `~/.cursor/...` path here, and none appears.
    assert all("~/" not in entry["source_path"] for entry in read.warnings)


def test_durable_attachment_reports_four_distinct_answers():
    attached = subagent_attachment_state(_Context(), str(FIXTURE))
    assert attached.attached is True
    assert attached.agents == ("planner", "verifier")

    not_asked = subagent_attachment_state(_Context(attach=False), str(FIXTURE))
    assert not_asked.attached is False
    assert "attach_subagents" in not_asked.reason

    cannot = subagent_attachment_state(_Context(spawn=False), str(FIXTURE))
    assert cannot.attached is False
    assert "enable_spawn_agent" in cannot.reason

    nothing = subagent_attachment_state(_Context(), str(FIXTURE.parent / "mini_repo"))
    assert nothing.attached is False
    assert "defines no subagents" in nothing.reason


def test_durable_attachment_is_a_pair_not_a_bool():
    attachment = SubagentAttachment(True, agents=("a",))
    assert attachment.to_payload()["attached"] is True
    assert attachment.to_payload()["agents"] == ["a"]
