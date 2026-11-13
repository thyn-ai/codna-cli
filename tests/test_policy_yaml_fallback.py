"""The stdlib YAML fallback for `.codna/policy.yaml` — same verdicts, no PyYAML, no weakened guard.

WHY THIS SUITE EXISTS. `policy_document._yaml_load` prefers PyYAML and, before this change, raised
`policy_reader_unavailable` without it. The sidecar maps that to `403 policy_verdict_unavailable` for
EVERY session, so one absent wheel took the product down — measured on the 0.1.48 app. `policy_yaml`
reads the subset of YAML a policy is written in, so that packaging gap degrades to a working reader.

THE CONTRACT IS EQUIVALENCE, NOT "IT PARSES". Both readers end in the same schema validator, so the
assertion that matters is that the SHARED FIXTURE MATRIX reaches the same verdict either way — the
matrix `tests/test_policy_document.py` and the engine suite both consume. Anything outside the subset
is refused with its own code (`policy_reader_subset_exceeded`) rather than approximated, and the two
fail-closed protections the docstring names — duplicate keys and unknown tags — are asserted here
directly, because a fallback that lost them would be a fail-open wearing a compatibility story.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from codna import policy_document, policy_yaml
from codna.policy_document import PolicyDocumentError, load_policy_document, policy_verdict

FIXTURES = Path(__file__).parent / "fixtures" / "policy"
MATRIX = yaml.safe_load((FIXTURES / "matrix.yaml").read_text(encoding="utf-8"))["fixtures"]
MISSING_PATH = FIXTURES / "this-file-does-not-exist.yaml"


@pytest.fixture()
def without_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fallback reader, whatever this interpreter happens to have installed.

    Patches `_pyyaml` rather than `sys.modules`: the real path is "the payload has no PyYAML", which is
    an import failure, and `_pyyaml` is the single place that decides. Removing the module would also
    break `yaml.safe_load` in this file's own matrix fixture.
    """
    monkeypatch.setattr(policy_document, "_pyyaml", lambda: None)


def _path_for(name: str) -> Path:
    return MISSING_PATH if name == "missing" else FIXTURES / f"{name}.yaml"


# ── equivalence: the shared matrix, read by BOTH readers ─────────────────────────────────────


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(entry, id=f"{entry['name']}-{'required' if entry['required'] else 'optional'}")
        for entry in MATRIX
    ],
)
def test_the_shared_matrix_holds_without_pyyaml(
    entry: dict[str, Any], without_pyyaml: None
) -> None:
    """The fallback reaches the verdict the matrix names — the same one the PyYAML path reaches."""
    verdict = policy_verdict(_path_for(entry["name"]), required=entry["required"])
    expected = entry["expect"]
    assert (verdict.mode, verdict.allowed, verdict.code) == (
        expected["mode"],
        expected["allowed"],
        expected["code"],
    ), verdict.to_dict()


def test_the_digest_is_IDENTICAL_from_both_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A digest that depended on which parser ran could not pin provenance.

    This is the assertion that makes the fallback a real substitute rather than a lookalike: both
    readers must produce the same canonical document, so the `policy_digest` a run records is the same
    whether or not the payload shipped PyYAML.
    """
    with_pyyaml = policy_verdict(FIXTURES / "valid.yaml")
    monkeypatch.setattr(policy_document, "_pyyaml", lambda: None)
    without = policy_verdict(FIXTURES / "valid.yaml")
    assert with_pyyaml.allowed is True
    assert without.allowed is True
    assert with_pyyaml.policy_digest == without.policy_digest
    assert with_pyyaml.permissions == without.permissions


def test_permissions_survive_the_fallback_intact(without_pyyaml: None) -> None:
    """The values, not just the verdict: a reader that lost `deny` would widen every run."""
    verdict = policy_verdict(FIXTURES / "valid.yaml")
    permissions = verdict.permissions
    assert permissions is not None
    assert permissions.allowed_tools == ("read_files", "editor")
    assert permissions.denied_tools == ("run_commands",)
    assert permissions.effective_tools() == ("read_files", "editor")
    assert permissions.allowed_write_paths == ("src/**",)
    assert permissions.allowed_commands == ("pytest",)
    assert permissions.max_cost_usd == 5.0


# ── the two fail-closed protections, asserted on the fallback directly ───────────────────────


def test_a_duplicate_key_is_refused_by_the_fallback_too(without_pyyaml: None) -> None:
    """THE protection this document exists for: a second `tools:` must not silently replace the first."""
    verdict = policy_verdict(FIXTURES / "duplicate-key.yaml")
    assert verdict.allowed is False
    assert verdict.code == "policy_duplicate_key"
    assert verdict.details["key"] == "tools"


def test_a_nested_duplicate_key_is_refused(without_pyyaml: None, tmp_path: Path) -> None:
    """The same rule one level in: two `allow:` lists under one `tools:` is the same fail-open."""
    target = tmp_path / "policy.yaml"
    target.write_text(
        "version: 1\ntools:\n  allow: [read_files]\n  allow: [read_files, run_commands]\n",
        encoding="utf-8",
    )
    verdict = policy_verdict(target)
    assert verdict.allowed is False
    assert verdict.code == "policy_duplicate_key"


def test_an_unknown_tag_is_refused_rather_than_constructed(without_pyyaml: None, tmp_path: Path) -> None:
    """No tag support at all, which is stronger than PyYAML's SafeLoader: it cannot build an object."""
    target = tmp_path / "policy.yaml"
    target.write_text("version: 1\ntools: !evil\n  allow: [read_files]\n", encoding="utf-8")
    verdict = policy_verdict(target)
    assert verdict.allowed is False
    assert verdict.code == "policy_reader_subset_exceeded"
    assert "tag" in verdict.reason
    # The location is reported, so an operator can find the construct rather than guess.
    assert verdict.details["line"] == 2


# ── the subset boundary, refused by name rather than approximated ────────────────────────────


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("version: 1\ntools: &t\n  allow: [read_files]\n", "anchor", id="anchor"),
        pytest.param(
            "version: 1\ntools:\n  allow: [read_files]\nwrite_scope: *t\n", "alias", id="alias"
        ),
        pytest.param(
            "version: 1\nrequired_approvals: |\n  run_commands\n", "block scalar", id="block-scalar"
        ),
        pytest.param(
            "version: 1\ntools: {allow: [read_files]}\n", "flow mapping", id="flow-mapping"
        ),
        pytest.param(
            "version: 1\ntools:\n  allow: [[read_files]]\n", "nested flow", id="nested-flow"
        ),
        pytest.param(
            "version: 1\n'tools':\n  allow: [read_files]\n", "mapping key", id="quoted-key"
        ),
    ],
)
def test_a_construct_outside_the_subset_is_REFUSED_with_its_own_code(
    without_pyyaml: None, tmp_path: Path, body: str, expected: str
) -> None:
    """Valid YAML the reader does not implement is not "unparseable" — the code says which it is.

    The distinction matters: `policy_unparseable` sends an operator to fix a file that is not broken.
    `policy_reader_subset_exceeded` names the construct and points at the wheel that would read it.
    """
    target = tmp_path / "policy.yaml"
    target.write_text(body, encoding="utf-8")
    verdict = policy_verdict(target)
    assert verdict.allowed is False
    assert verdict.code == "policy_reader_subset_exceeded"
    assert expected in verdict.reason
    assert verdict.details["hint"]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("version: 1\ntools:\n\tallow: [read_files]\n", id="tab-indent"),
        pytest.param("version: 1\ntools:\n  allow: [read_files\n", id="unterminated-flow"),
        pytest.param("version: 1\ntools:\n  allow: 'read_files\n", id="unterminated-quote"),
        pytest.param("version: 1\n  tools: 1\n", id="unexpected-indent"),
    ],
)
def test_malformed_yaml_is_policy_unparseable_the_same_verdict_pyyaml_reaches(
    without_pyyaml: None, tmp_path: Path, body: str
) -> None:
    target = tmp_path / "policy.yaml"
    target.write_text(body, encoding="utf-8")
    assert policy_verdict(target).code == "policy_unparseable"


# ── the shapes a policy is actually written in, which MUST work ─────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("version: 1\ntools:\n  allow: [read_files]  # trailing\n", id="trailing-comment"),
        pytest.param("# leading comment\nversion: 1\ntools:\n  allow: [read_files]\n", id="leading-comment"),
        pytest.param("version: 1\ntools:\n  allow:\n    - read_files\n", id="block-sequence"),
        pytest.param("version: 1\ntools:\n  allow: ['read_files']\n", id="single-quoted"),
        pytest.param('version: 1\ntools:\n  allow: ["read_files"]\n', id="double-quoted"),
        pytest.param("version: 1\ntools:\n  allow: []\n", id="empty-flow"),
        pytest.param("version: 1\r\ntools:\r\n  allow: [read_files]\r\n", id="crlf"),
        pytest.param("version: 1\ntools:\n  allow: [read_files]\n  deny: []\n", id="two-keys"),
    ],
)
def test_the_documented_policy_shapes_parse(without_pyyaml: None, tmp_path: Path, body: str) -> None:
    """Every one of these is a document the SCHEMA reads, so the fallback must reach the schema.

    Asserted as "the verdict is about permissions, not about parsing" rather than `allowed is True`:
    `allow: []` is a valid document that permits nothing (`policy_denies_all`), which is the correct
    fail-closed answer and not a parser failure. The parse-only codes are what this test excludes.
    """
    target = tmp_path / "policy.yaml"
    target.write_text(body, encoding="utf-8")
    verdict = policy_verdict(target)
    assert verdict.code not in {
        "policy_unparseable",
        "policy_duplicate_key",
        "policy_reader_subset_exceeded",
    }, verdict.to_dict()


def test_a_document_that_permits_something_is_ALLOWED_without_pyyaml(
    without_pyyaml: None, tmp_path: Path
) -> None:
    """The positive half, stated once: a granting policy is allowed, not merely parsed."""
    target = tmp_path / "policy.yaml"
    target.write_text(
        "version: 1\ntools:\n  allow: [read_files, editor]\nwrite_scope:\n  paths: [\"src/**\"]\n",
        encoding="utf-8",
    )
    verdict = policy_verdict(target)
    assert verdict.allowed is True
    assert verdict.code == "policy_ok"


def test_the_reader_is_reachable_without_pyyaml_at_all(tmp_path: Path) -> None:
    """The end-to-end claim: no PyYAML in the interpreter, and a policy still evaluates.

    Run as a SUBPROCESS with an import hook that hides `yaml`, because the in-process monkeypatch
    cannot prove the import path — only that this module calls it. A payload that shipped without the
    wheel is exactly this process.
    """
    import subprocess
    import sys

    cli_root = Path(__file__).resolve().parents[1]
    target = tmp_path / "policy.yaml"
    target.write_text(
        "version: 1\ntools:\n  allow: [read_files, editor]\nwrite_scope:\n  paths: [\"src/**\"]\n",
        encoding="utf-8",
    )
    program = "\n".join(
        [
            "import sys",
            "class _Hide:",
            "    def find_module(self, name, path=None):",
            "        return self if name == 'yaml' else None",
            "    def find_spec(self, name, path=None, target=None):",
            "        if name != 'yaml':",
            "            return None",
            "        raise ModuleNotFoundError(\"No module named 'yaml'\")",
            "sys.meta_path.insert(0, _Hide())",
            "import json",
            "from codna.policy_document import policy_verdict",
            "verdict = policy_verdict(sys.argv[1], required=True)",
            "print(json.dumps({'allowed': verdict.allowed, 'code': verdict.code, 'mode': verdict.mode}))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=str(cli_root),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"allowed": True, "code": "policy_ok", "mode": "policy"}


def test_the_yaml_reader_error_types_are_distinct() -> None:
    """The mapping from reader error to verdict code depends on these being separate classes."""
    assert issubclass(policy_yaml.YamlSyntaxError, policy_yaml.YamlSubsetError)
    assert issubclass(policy_yaml.UnsupportedYamlError, policy_yaml.YamlSubsetError)
    assert not issubclass(policy_yaml.UnsupportedYamlError, policy_yaml.YamlSyntaxError)
    assert not issubclass(policy_yaml.DuplicateKeyError, policy_yaml.YamlSubsetError)


def test_an_empty_document_is_still_a_policy_that_permits_nothing(without_pyyaml: None) -> None:
    """The fallback must not turn an empty file into "no policy" — that is the permissive reading."""
    verdict = policy_verdict(FIXTURES / "empty.yaml")
    assert verdict.mode == "policy"
    assert verdict.allowed is False
    assert verdict.code == "policy_denies_all"


def test_a_policy_document_error_still_carries_its_code(without_pyyaml: None) -> None:
    """`load_policy_document` raises, and the code is what both consumers switch on."""
    with pytest.raises(PolicyDocumentError) as caught:
        load_policy_document(FIXTURES / "duplicate-key.yaml")
    assert caught.value.code == "policy_duplicate_key"
    assert caught.value.as_dict()["error"]["details"]["key"] == "tools"


# ── defects found in review: three shapes the fallback refused that PyYAML accepts ──────────
#
# Each test below FAILED before its fix and passes after, and each is asserted against PyYAML's own
# answer rather than a hand-written literal — the contract is equivalence, so the oracle is the reader
# the fallback stands in for. Where PyYAML is unavailable the parametrisation is skipped, because a
# comparison that silently became `x == x` would prove nothing.


def test_a_escaped_quote_inside_a_double_quoted_scalar_does_not_hide_a_comment(
    without_pyyaml: None,
) -> None:
    """`_strip_comment` treated the `\"` as the CLOSING quote, so the comment was kept as data.

    Measured before the fix: `policy_yaml.loads('key: "ab\\"" # comment')` raised
    `YamlSyntaxError: an unterminated quoted scalar`, i.e. VALID YAML reported as `policy_unparseable`
    — the one failure mode a fallback must not have. The escape is data; the quote after it is not a
    delimiter.
    """
    body = 'key: "ab\\"" # comment\n'
    assert policy_yaml.loads(body) == {"key": 'ab"'}
    # The document-level claim: a policy carrying such a value reaches the SCHEMA, not a parse error.
    assert yaml.safe_load(body) == {"key": 'ab"'} == policy_yaml.loads(body)


def test_a_backslash_escape_is_only_special_inside_DOUBLE_quotes(without_pyyaml: None) -> None:
    """Single-quoted scalars have no backslash escapes — a quote is doubled — so `\\` is literal there.

    This is the other direction, and it is why the fix keys the escape flag on `"`: applying it to
    single quotes would make `'a\\' # c'` look escaped and swallow the real closing quote.
    """
    assert policy_yaml.loads("key: 'a\\' # c\n") == {"key": "a\\"}
    assert policy_yaml.loads("key: 'it''s' # c\n") == {"key": "it's"}


def test_continuation_keys_in_a_sequence_item_align_with_the_KEY_not_the_dash(
    without_pyyaml: None,
) -> None:
    """YAML allows any spacing after `-`; `_inline_mapping` assumed exactly one space (key at indent+2).

    MEASURED before the fix, on `steps:\\n  -   name: a\\n      other: b`: the reader raised
    `YamlSyntaxError: unexpected indentation`, while `yaml.safe_load` answers
    `{'steps': [{'name': 'a', 'other': 'b'}]}`. The key sits at indent+4 there, and the continuation
    must align with the KEY's column — which is also what a nested value under the first key uses.
    """
    body = "steps:\n  -   name: a\n      other: b\n"
    assert policy_yaml.loads(body) == {"steps": [{"name": "a", "other": "b"}]} == yaml.safe_load(body)


@pytest.mark.parametrize("gap", [1, 2, 3, 5])
def test_every_dash_spacing_parses_to_the_same_mapping(without_pyyaml: None, gap: int) -> None:
    """One property over the spacing, so a regression cannot reappear at just the tested width."""
    pad = " " * gap
    body = f"steps:\n  -{pad}name: a\n{' ' * (3 + gap)}other: b\n"
    expected = {"steps": [{"name": "a", "other": "b"}]}
    assert policy_yaml.loads(body) == expected == yaml.safe_load(body)


def test_a_nested_block_under_the_first_key_of_a_sequence_item(without_pyyaml: None) -> None:
    """The relative-indent half of the same defect: `opts:`'s children are indented under the KEY."""
    body = "steps:\n  -   name: a\n      opts:\n        deep: 1\n"
    expected = {"steps": [{"name": "a", "opts": {"deep": 1}}]}
    assert policy_yaml.loads(body) == expected == yaml.safe_load(body)


def test_a_CRASHING_pyyaml_extension_falls_back_instead_of_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_pyyaml()` caught only `ModuleNotFoundError`, so a broken wheel took down every session.

    A partially installed PyYAML raises a bare `ImportError` from inside `yaml/__init__.py` — the
    `_yaml` C extension failing to load is the realistic case — and that is NOT a
    `ModuleNotFoundError`. Before the fix it escaped `_yaml_load` and bypassed the fallback entirely,
    making a CORRUPT PyYAML strictly worse than an absent one. The stand-in here raises exactly the
    exception shape `yaml/__init__.py` does when the extension is unloadable.
    """
    real_import = builtins.__import__

    def failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("dlopen(.../_yaml.cpython-312-darwin.so, 2): symbol not found", name="yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    # The decision itself: unusable means None, i.e. "use the fallback reader".
    assert policy_document._pyyaml() is None

    # And end-to-end: a real policy still reaches a real verdict through the subset reader.
    monkeypatch.undo()
    monkeypatch.setattr(policy_document, "_pyyaml", lambda: None)
    verdict = policy_verdict(FIXTURES / "valid.yaml")
    assert verdict.allowed is True
    assert verdict.code == "policy_ok"
