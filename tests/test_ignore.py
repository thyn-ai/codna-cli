"""Gitignore-semantics proof for ``codna.ignore``.

The matcher is a hand-written port, so the semantics are PINNED rather than asserted: the table
below is git's own documented pattern table (``gitignore(5)`` "PATTERN FORMAT" / "NOTES"), and
every entry is additionally cross-checked against a real ``git check-ignore`` run, so a future
"cleanup" of the translator cannot quietly diverge from git.

The `data/` + `!data/keep.py` case is deliberately in the table: git documents that a file
cannot be re-included when a parent DIRECTORY is excluded, which is the rule the whole feature's
security argument rests on (see `test_negation_cannot_reinclude_secret`).
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from codna.ignore import IgnoreStack, compile_rule, parse_patterns

# (pattern, path, expected-ignored) — git's documented table plus the anchoring/negation cases.
SEMANTICS_TABLE: tuple[tuple[str, str, bool], ...] = (
    # `*` matches anything EXCEPT a slash.
    ("*.log", "a.log", True),
    ("*.log", "x/a.log", True),
    ("*.log", "x/a.txt", False),
    # `?` matches one character except a slash.
    ("f?o", "foo", True),
    ("f?o", "f/o", False),
    # Character classes, including `[!...]` negation.
    ("[abc].py", "a.py", True),
    ("[abc].py", "d.py", False),
    ("[!abc].py", "d.py", True),
    ("[!abc].py", "a.py", False),
    # `**` crosses directories; `**/` matches zero or more leading directories.
    ("a/**/b", "a/b", True),
    ("a/**/b", "a/x/y/b", True),
    ("a/**/b", "x/a/b", False),
    ("**/foo", "foo", True),
    ("**/foo", "x/foo", True),
    ("bar/**", "bar/x", True),
    ("bar/**", "x/bar/x", False),
    # A leading slash anchors to the ignore file's directory; a middle slash also anchors.
    ("/rooted", "rooted", True),
    ("/rooted", "sub/rooted", False),
    ("doc/frotz", "doc/frotz", True),
    ("doc/frotz", "doc/x/frotz", False),
    # A trailing slash means DIRECTORY ONLY, and covers the whole subtree.
    ("dir/", "dir", True),
    ("dir/", "dir/x", True),
    ("dir/", "sub/dir/x", True),
    ("dir/", "dir", True),
    # A name with no slash floats to any depth, and covers the DIRECTORY's subtree (git).
    ("data", "data", True),
    ("data", "a/data/x", True),
    ("data", "database", False),
)

# Patterns whose NOTHING under a directory is re-included, per gitignore(5) NOTES.
NESTED_NEGATION_TABLE: tuple[tuple[str, str, bool], ...] = (
    # "It is not possible to re-include a file if a parent directory of that file is excluded."
    ("data/\n!data/keep.py", "data/keep.py", True),
    ("data/\n!data/keep.py", "data/sub/keep.py", True),
    # Excluding the CONTENTS of a directory DOES allow a specific file back in.
    ("data/*\n!data/keep.py", "data/keep.py", False),
    ("data/*\n!data/keep.py", "data/drop.py", True),
)


def _rules(lines: list[str]):
    return parse_patterns(lines, base="", source=".gitignore")


@pytest.mark.parametrize("pattern,path,expected", SEMANTICS_TABLE)
def test_gitignore_semantics_table(pattern: str, path: str, expected: bool):
    """gitignore's documented pattern table, compiled and matched one rule at a time."""
    rule = compile_rule(pattern, base="", source=".gitignore", line=1)
    assert rule is not None, pattern
    assert rule.matches(path) is expected, f"{pattern!r} vs {path!r}"


def test_last_match_wins():
    """gitignore's rule: the LAST matching pattern decides, not the first."""
    rules = _rules(["*.log", "!keep.log"])
    assert any(r.matches("keep.log") and r.negated for r in rules)

    ignore_lines = ["*.log", "!keep.log"]
    layer = parse_patterns(ignore_lines, base="", source=".gitignore")
    decision: bool | None = None
    for rule in layer:
        if rule.matches("keep.log"):
            decision = not rule.negated
    assert decision is False  # the negation came later, so it wins

    # The reverse order means the exclusion wins, even though a negation appeared first.
    layer = parse_patterns(["!keep.log", "*.log"], base="", source=".gitignore")
    decision = None
    for rule in layer:
        if rule.matches("keep.log"):
            decision = not rule.negated
    assert decision is True


def test_comments_and_blank_lines_are_not_rules():
    rules = parse_patterns(["# a comment", "", "   ", "real.log"], base="", source=".gitignore")
    assert [rule.pattern for rule in rules] == ["real.log"]


def test_escaped_hash_is_a_literal_pattern():
    rule = compile_rule("\\#literal", base="", source=".gitignore", line=1)
    assert rule is not None and rule.matches("#literal")


def test_trailing_spaces_are_ignored_unless_escaped():
    stripped = compile_rule("name   ", base="", source=".gitignore", line=1)
    assert stripped is not None
    assert stripped.matches("name") and stripped.matches("x/name")
    assert not stripped.matches("name   ")  # the spaces were stripped, not part of the name
    kept = compile_rule("name\\ ", base="", source=".gitignore", line=1)
    assert kept is not None and kept.matches("name ")


# ── cross-check against real git ────────────────────────────────────────────────────────────
# The table above encodes what the docs SAY. This proves the SAME answers come out of a real
# git binary, so a mistaken reading of the docs cannot make both the code and its test wrong in
# the same direction. Skipped only where git is unavailable (no project.skip on this repo).
HAS_GIT = shutil.which("git") is not None


@pytest.mark.skipif(not HAS_GIT, reason="git binary not available to cross-check semantics")
def test_semantics_match_real_git(tmp_path):
    """Every single-pattern row agrees with `git check-ignore`.

    Rows whose verdict depends on the PARENT-DIRECTORY rule are covered by
    `test_parent_directory_exclusion_blocks_reinclusion` instead, because they need two patterns.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    mismatches: list[str] = []
    for pattern, path, expected in SEMANTICS_TABLE:
        (tmp_path / ".gitignore").write_text(pattern + "\n", encoding="utf-8")
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("", encoding="utf-8")
        # A DIRECTORY-ONLY row needs the path to BE a directory for git to agree.
        if pattern.endswith("/") and target.is_file():
            target.unlink()
            target.mkdir(parents=True, exist_ok=True)
        probe = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", "--", path],
            cwd=tmp_path,
            check=False,
        )
        git_says = probe.returncode == 0
        if git_says != expected:
            mismatches.append(f"{pattern!r} {path!r}: table={expected} git={git_says}")
        if target.is_file():
            target.unlink()
    assert not mismatches, "table diverges from real git:\n" + "\n".join(mismatches)


@pytest.mark.skipif(not HAS_GIT, reason="git binary not available to cross-check semantics")
@pytest.mark.parametrize("patterns,path,expected_ignored", NESTED_NEGATION_TABLE)
def test_parent_directory_exclusion_blocks_reinclusion(tmp_path, patterns, path, expected_ignored):
    """The parent-directory rule, cross-checked against real git."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(patterns + "\n", encoding="utf-8")
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    probe = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", path], cwd=tmp_path, check=False
    )
    git_says = probe.returncode == 0
    assert git_says == expected_ignored, f"{patterns!r} {path!r}: git={git_says}"

    # And the matcher agrees with git, through the stack (floor disabled so only the file decides).
    (tmp_path / ".gitignore").write_text(patterns + "\n", encoding="utf-8")
    stack = IgnoreStack.build(tmp_path, floor=())
    assert stack.is_ignored(path) is expected_ignored


# ── the stack: precedence + the floor ───────────────────────────────────────────────────────
def _write(root, relpath: str, text: str) -> None:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_negation_cannot_reinclude_secret(tmp_path):
    """A `.codnaignore` saying `!.env` must NOT re-expose a secret: the floor is a floor."""
    _write(tmp_path, ".env", "SECRET=1")
    _write(tmp_path, ".codnaignore", "!.env\n!id_rsa\n!secrets/\n!.env.local\n")
    stack = IgnoreStack.build(tmp_path)

    assert stack.is_ignored(".env") is True
    assert stack.is_ignored("id_rsa") is True
    assert stack.is_ignored("secrets/token") is True
    assert stack.is_ignored(".env.local") is True


def test_floor_cannot_be_weakened_for_heavy_dirs(tmp_path):
    _write(tmp_path, "node_modules/x/index.js", "")
    _write(tmp_path, ".codnaignore", "!node_modules/\n!build/\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored("node_modules/x/index.js") is True
    assert stack.is_ignored("build/out.js") is True


def test_user_files_apply_on_top_of_the_floor(tmp_path):
    _write(tmp_path, ".codnaignore", "vendor/\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored("vendor/lib.py") is True
    # A path the user did NOT exclude is still included — the ignore set is additive.
    assert stack.is_ignored("src/app.py") is False


def test_last_match_wins_across_layers(tmp_path):
    """`.gitignore` then `.cursorignore` then `.codnaignore`; the Codna file decides last."""
    _write(tmp_path, ".gitignore", "data/\n")
    _write(tmp_path, ".cursorignore", "fixtures/\n")
    _write(tmp_path, ".codnaignore", "!fixtures/\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored("data/x.py") is True
    assert stack.is_ignored("fixtures/y.py") is False  # the Codna layer re-included it


def test_codnaignore_wins_over_gitignore(tmp_path):
    _write(tmp_path, ".gitignore", "generated/\n")
    _write(tmp_path, ".codnaignore", "!generated/\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored("generated/a.py") is False


def test_nested_ignore_scopes_to_subtree(tmp_path):
    """A nested file applies to its own subtree only — siblings are unaffected."""
    _write(tmp_path, "svc-a/.codnaignore", "secret_config.py\n")
    _write(tmp_path, "svc-b/secret_config.py", "")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored("svc-a/secret_config.py") is True
    assert stack.is_ignored("svc-b/secret_config.py") is False
    assert stack.is_ignored("secret_config.py") is False


def test_nested_rules_are_relative_to_their_directory(tmp_path):
    """`/local.py` in `svc/` anchors to `svc/`, not to the repository root."""
    _write(tmp_path, "svc/.codnaignore", "/local.py\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored("svc/local.py") is True
    assert stack.is_ignored("local.py") is False


def test_deeper_file_overrides_shallower_for_its_subtree(tmp_path):
    _write(tmp_path, ".codnaignore", "*.log\n")
    _write(tmp_path, "keep/.codnaignore", "!*.log\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored("other/app.log") is True
    assert stack.is_ignored("keep/app.log") is False


def test_include_gitignore_false_drops_only_the_implicit_layer(tmp_path):
    _write(tmp_path, ".gitignore", "data/\n")
    _write(tmp_path, ".codnaignore", "vendor/\n")
    stack = IgnoreStack.build(tmp_path, include_gitignore=False)
    assert stack.is_ignored("data/x.py") is False   # the implicit layer was dropped
    assert stack.is_ignored("vendor/x.py") is True  # the explicit one was not
    assert stack.is_ignored(".env") is True         # the floor is never dropped


def test_hidden_directories_are_floored_but_dot_files_are_not(tmp_path):
    """Matches the old walker exactly: hidden DIRS pruned, dot-FILES still indexable."""
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_ignored(".github/workflows/ci.yml") is True
    assert stack.is_ignored_dir(".github") is True
    assert stack.is_ignored(".eslintrc.js") is False


def test_empty_repository_has_no_rules(tmp_path):
    stack = IgnoreStack.build(tmp_path)
    assert stack.sources() == ()
    assert stack.is_ignored("src/app.py") is False


# ── the indexer, end to end ────────────────────────────────────────────────────────────────
# These are the user-visible defect: units must not be produced for paths a repo excludes.
def _repo(tmp_path, files: dict[str, str]) -> str:
    root = str(tmp_path / "repo")
    for rel, text in files.items():
        target = tmp_path / "repo" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def test_gitignored_dir_not_indexed(tmp_path):
    """The headline fix: a `data/` in `.gitignore` yields zero units."""
    from codna.codeunits import extract_repo

    root = _repo(tmp_path, {
        ".gitignore": "data/\n",
        "app.py": "def keep():\n    return 1\n",
        "data/loader.py": "def must_not_be_indexed():\n    return 2\n",
        "data/nested/deep.py": "def also_excluded():\n    return 3\n",
    })
    units, stats = extract_repo(root, "r", languages=("python",))
    paths = {unit.path for unit in units}
    assert "app.py" in paths
    assert not any(path.startswith("data/") for path in paths), paths
    assert stats.files == 1


def test_cursorignore_is_honoured(tmp_path):
    """A Cursor repo's own `.cursorignore` works with zero changes."""
    from codna.codeunits import extract_repo

    root = _repo(tmp_path, {
        ".cursorignore": "fixtures/\n",
        "app.py": "def keep():\n    return 1\n",
        "fixtures/sample.py": "def excluded():\n    return 2\n",
    })
    units, _ = extract_repo(root, "r", languages=("python",))
    paths = {unit.path for unit in units}
    assert "app.py" in paths
    assert not any(path.startswith("fixtures/") for path in paths), paths


def test_negation_cannot_reinclude_secret_in_the_index(tmp_path):
    """A `!.env` in `.codnaignore` cannot make the indexer embed `.env`."""
    from codna.codeunits import extract_repo

    root = _repo(tmp_path, {
        ".codnaignore": "!.env\n!secrets/\n",
        "app.py": "def keep():\n    return 1\n",
        "secrets/token.py": "def leak():\n    return 2\n",
    })
    (tmp_path / "repo" / "secrets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / ".env").write_text("SECRET=1\n", encoding="utf-8")
    units, _ = extract_repo(root, "r", languages=("python",))
    paths = {unit.path for unit in units}
    assert paths == {"app.py"}, paths


def test_indexer_still_indexes_a_repo_with_no_ignore_files(tmp_path):
    """The documented migration exception: only a `.gitignore` changes behaviour, nothing else.

    In particular a DOT-FILE is still indexed: the old walker pruned hidden DIRECTORIES but never
    filtered files by a leading dot, and this change must not quietly start excluding them.
    """
    from codna.codeunits import extract_repo

    root = _repo(tmp_path, {
        "app.py": "def keep():\n    return 1\n",
        "lib/util.py": "def helper():\n    return 2\n",
        ".eslintrc.py": "def dotfile_symbol():\n    return 3\n",
    })
    units, _ = extract_repo(root, "r", languages=("python",))
    paths = {unit.path for unit in units}
    assert {"app.py", "lib/util.py", ".eslintrc.py"} <= paths


# ── regression pins for the review findings on PR #714 ────────────────────────────────────────
def test_normalization_is_linear_and_handles_repeated_dot_prefixes(tmp_path):
    """No regex backtracking, and `./././x` normalizes like `x` (the IDE matcher agrees)."""
    _write(tmp_path, ".codnaignore", "data/\n")
    stack = IgnoreStack.build(tmp_path)

    assert stack.is_ignored("./" * 10_000 + "src/app.py") is False
    assert stack.is_ignored("/" * 100_000) is False
    # A repeated "./" prefix must not leave a "." segment behind (which would look like a hidden dir).
    assert stack.is_ignored("./././src/app.py") is False
    assert stack.is_ignored("././data/x.py") is True


def test_dir_only_pattern_does_not_match_a_file_of_the_same_name(tmp_path):
    """A trailing-slash pattern is DIRECTORY-ONLY (git: a file named `data` is not excluded)."""
    _write(tmp_path, ".gitignore", "data/\n")
    stack = IgnoreStack.build(tmp_path, floor=())
    assert stack.is_ignored("data") is False        # the FILE named data
    assert stack.is_ignored("data/x.py") is True    # the directory's contents


def test_floor_directories_are_pruned_even_with_a_negation_elsewhere(tmp_path):
    """`!keep.log` must not make the walk descend `node_modules/` file by file."""
    _write(tmp_path, ".gitignore", "!keep.log\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_floor_ignored_dir("node_modules") is True
    assert stack.is_floor_ignored_dir("node_modules/pkg/dist") is True
    assert stack.is_floor_ignored_dir(".git") is True
    # A user-excluded directory is NOT a floor directory — a negation could re-include under it.
    _write(tmp_path, ".codnaignore", "data/\n")
    stack = IgnoreStack.build(tmp_path)
    assert stack.is_floor_ignored_dir("data") is False
    assert stack.is_ignored_dir("data") is True


def test_negation_still_reaches_into_a_user_excluded_directory(tmp_path):
    """The pruning exemption for user dirs is load-bearing: this must survive a walk."""
    from codna.codeunits import extract_repo

    root = _repo(tmp_path, {
        ".codnaignore": "data/*\n!data/keep.py\n",
        "app.py": "def keep():\n    return 1\n",
        "data/keep.py": "def re_included():\n    return 2\n",
        "data/drop.py": "def dropped():\n    return 3\n",
    })
    units, _ = extract_repo(root, "r", languages=("python",))
    paths = {unit.path for unit in units}
    assert "data/keep.py" in paths, paths
    assert "data/drop.py" not in paths, paths


def test_floor_directories_are_not_walked_even_with_negations(tmp_path):
    """The indexer must not descend a floored tree; the file count proves it was never visited."""
    from codna.codeunits import extract_repo

    root = _repo(tmp_path, {
        ".gitignore": "!keep.log\n",
        "app.py": "def keep():\n    return 1\n",
        "node_modules/pkg/huge.py": "def dependency_code():\n    return 2\n",
    })
    units, stats = extract_repo(root, "r", languages=("python",))
    paths = {unit.path for unit in units}
    assert paths == {"app.py"}, paths
    assert stats.files == 1
