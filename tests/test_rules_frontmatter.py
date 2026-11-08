"""The PyYAML-absent fallback parser: honest, or it refuses.

``frontmatter._fallback_parse`` runs only on a machine without PyYAML, which is why it is worth its own
tests: an environment-dependent code path that silently mis-parses a rule is a rule that stops applying
on that machine and nowhere else. Every case here is either "agrees with PyYAML" or "refuses with
:class:`FrontmatterError`" — there is no third, silently-different answer.

The three cases at the top came from a real review of this file; each was confirmed by execution before
being fixed, so they are regression tests rather than speculation.
"""
from __future__ import annotations

import pytest

from codna.rules import frontmatter
from codna.rules.frontmatter import FrontmatterError, parse_markdown


def fallback(header: str) -> dict:
    """Parse with the fallback ONLY, regardless of whether PyYAML is installed here."""
    return frontmatter._fallback_parse(header, source_path="test.mdc")


def yaml_reading(header: str):
    """PyYAML's own reading, so the two can be compared rather than assumed equal."""
    import yaml

    return yaml.safe_load(header)


# ---------------------------------------------------------------------------------------------
# the three review findings, each reproduced before being fixed
# ---------------------------------------------------------------------------------------------

def test_a_flow_list_keeps_brace_alternation_intact():
    """A comma inside ``{}`` belongs to the group, not to the list.

    A naive ``inner.split(",")`` turned ``[{src,tests}/**/*.ts, docs/**]`` into three patterns that
    match nothing, so a glob-scoped rule silently stopped applying on a machine without PyYAML.
    """
    parsed = fallback("globs: [{src,tests}/**/*.ts, docs/**]")
    assert parsed["globs"] == ["{src,tests}/**/*.ts", "docs/**"]


def test_a_mismatched_quote_is_a_FrontmatterError_not_a_JSONDecodeError():
    """``JSONDecodeError`` is not a ``FrontmatterError``, so it would escape discovery into a caller.

    Discovery catches ``FrontmatterError``; anything else travels out of ``discover`` → ``load`` →
    ``guidance_for_repo`` and crashes a turn. The type matters as much as the refusal.
    """
    with pytest.raises(FrontmatterError):
        fallback('description: "hello\'')


def test_a_bare_block_list_key_reads_as_None_like_PyYAML():
    """``globs:`` with no items is ``None``; dropping the key would make the parsers disagree."""
    header = "globs:\ndescription: foo\n"
    assert fallback(header) == {"globs": None, "description": "foo"}
    assert fallback(header) == yaml_reading(header)


def test_an_unterminated_flow_list_is_refused():
    """One pattern that matches nothing is worse than an error: it looks applied and is not."""
    with pytest.raises(FrontmatterError):
        fallback('globs: ["src/**')


# ---------------------------------------------------------------------------------------------
# agreement with PyYAML, where both must read the same file identically
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "header",
    [
        "alwaysApply: true",
        "alwaysApply: false",
        "description: simple text",
        "description: 'single quoted'",
        'description: "double quoted"',
        "description: has a # not a comment?",
        "n: 42",
        "nothing: null",
        "tilde: ~",
        "globs:\n  - a/**\n  - b/**",
        "globs: [a/**, b/**]",
        "globs: []",
        "a: 1\nb: 2",
    ],
)
def test_the_fallback_agrees_with_pyyaml(header):
    assert fallback(header) == yaml_reading(header), header


def test_single_quoted_doubled_quotes_are_unescaped():
    """YAML's own single-quote escape (``''`` → ``'``) must survive the fallback."""
    assert fallback("description: 'it''s here'") == {"description": "it's here"}


def test_a_trailing_comment_is_stripped_but_a_hash_inside_quotes_is_not():
    assert fallback("description: value # comment") == {"description": "value"}
    assert fallback('description: "a # b" # comment') == {"description": "a # b"}


# ---------------------------------------------------------------------------------------------
# refusals: shapes outside the documented subset must not be guessed at
# ---------------------------------------------------------------------------------------------

def test_a_nested_mapping_is_refused_rather_than_flattened():
    with pytest.raises(FrontmatterError):
        fallback("meta:\n  nested: value\n")


def test_a_list_item_without_a_key_is_refused():
    with pytest.raises(FrontmatterError):
        fallback("- orphan\n")


# ---------------------------------------------------------------------------------------------
# through parse_markdown: the frontmatter/body split and the had_frontmatter flag
# ---------------------------------------------------------------------------------------------

def test_parse_markdown_splits_header_from_body():
    metadata, body, had = parse_markdown(
        "---\nalwaysApply: true\n---\nthe body\n", source_path="x.mdc")
    assert metadata == {"alwaysApply": True}
    assert body == "the body\n"
    assert had is True


def test_parse_markdown_without_a_fence_returns_the_whole_file():
    metadata, body, had = parse_markdown("just a rule\n", source_path="x.mdc")
    assert metadata == {}
    assert body == "just a rule\n"
    assert had is False


def test_parse_markdown_accepts_plus_fences():
    metadata, body, had = parse_markdown("+++\nalwaysApply: true\n+++\nbody\n", source_path="x.mdc")
    assert metadata == {"alwaysApply": True}
    assert had is True
    assert body == "body\n"


def test_parse_markdown_treats_an_unterminated_fence_as_body():
    """An opening fence with no close is not frontmatter; the file is body, and discovery reports it."""
    metadata, body, had = parse_markdown("---\nalwaysApply: true\n", source_path="x.mdc")
    assert metadata == {}
    assert had is False
    assert "alwaysApply" in body


def test_parse_markdown_reports_a_non_mapping_header():
    with pytest.raises(FrontmatterError):
        parse_markdown("---\n- a list\n- not a mapping\n---\nbody\n", source_path="x.mdc")


def test_a_bom_is_tolerated():
    metadata, _body, had = parse_markdown(
        "\ufeff---\nalwaysApply: true\n---\nbody\n", source_path="x.mdc")
    assert had is True
    assert metadata == {"alwaysApply": True}


def test_frontmatter_errors_name_the_source_path():
    """A fallback-only error names the file, so a user knows which rule to fix.

    Uses a shape only the FALLBACK can hit (an unterminated flow list), because that is where the
    fallback's message is the one a user sees. With PyYAML installed the parse is PyYAML's and its
    error is already wrapped by `parse_markdown` with the same path prefix.
    """
    with pytest.raises(FrontmatterError) as excinfo:
        frontmatter._fallback_parse('globs: ["src/**', source_path=".codna/rules/x.mdc")
    assert ".codna/rules/x.mdc" in str(excinfo.value)


def test_a_non_boolean_alwaysApply_is_refused():
    """`always_apply` is co-erced nowhere: discovery raises on a non-bool, and this pins that."""
    from codna.rules.discovery import _apply_frontmatter
    from codna.rules.model import Rule, Scope

    rule = Rule(name="x", body="b", source_path="x.mdc", scope=Scope.TEAM)
    with pytest.raises(ValueError):
        _apply_frontmatter(rule, {"alwaysApply": "yes"}, in_rules_dir=True)
