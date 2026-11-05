"""Policy-as-code: `.codna/policy.yaml` MUST fail closed, and both consumers must agree.

The fixture matrix in `tests/fixtures/policy/matrix.yaml` is the contract. This suite asserts the
PYTHON reader (`codna.policy_document`) reaches every verdict in it; the engine suite
(`algenta/server/policy-document.test.ts`) asserts the SIDECAR reaches the same verdicts from the
same files. Neither side may special-case a fixture — a disagreement is a failure on both.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from codna.policy_document import (
    PolicyDocumentError,
    load_policy_document,
    main,
    policy_path_for,
    policy_verdict,
)

FIXTURES = Path(__file__).parent / "fixtures" / "policy"
MATRIX = yaml.safe_load((FIXTURES / "matrix.yaml").read_text(encoding="utf-8"))["fixtures"]
MISSING_PATH = FIXTURES / "this-file-does-not-exist.yaml"


def _cases() -> list:
    return [
        pytest.param(entry, id=f"{entry['name']}-{'required' if entry['required'] else 'optional'}")
        for entry in MATRIX
    ]


def _path_for(name: str) -> Path:
    return MISSING_PATH if name == "missing" else FIXTURES / f"{name}.yaml"


@pytest.mark.parametrize("entry", _cases())
def test_the_shared_matrix_holds(entry: dict) -> None:
    """Every fixture reaches the verdict the matrix names. This is the two-consumer contract."""
    verdict = policy_verdict(_path_for(entry["name"]), required=entry["required"])
    expected = entry["expect"]
    assert (verdict.mode, verdict.allowed, verdict.code) == (
        expected["mode"],
        expected["allowed"],
        expected["code"],
    ), verdict.to_dict()


def test_an_empty_file_is_a_policy_that_permits_nothing() -> None:
    verdict = policy_verdict(FIXTURES / "empty.yaml")
    assert verdict.allowed is False
    assert verdict.code == "policy_denies_all"
    assert verdict.permissions is None, "a denying verdict must not carry permissions to apply"


def test_an_unknown_key_denies_rather_than_being_ignored() -> None:
    """THE fail-open this schema exists to prevent: `budjets:` typo'd, everything else permissive."""
    verdict = policy_verdict(FIXTURES / "permissive-looking-invalid.yaml")
    assert verdict.allowed is False
    assert verdict.code == "policy_unknown_key"
    assert "budjets" in verdict.details["unknown"]


def test_a_duplicate_key_is_a_parse_error_not_last_wins() -> None:
    """PyYAML's default loader keeps the LAST `tools:` block. That is a fail-open, so it is refused."""
    verdict = policy_verdict(FIXTURES / "duplicate-key.yaml")
    assert verdict.code == "policy_duplicate_key"


def test_a_missing_file_denies_when_a_policy_is_required() -> None:
    verdict = policy_verdict(MISSING_PATH, required=True)
    assert verdict.mode == "policy"
    assert verdict.allowed is False
    assert verdict.code == "policy_missing"


def test_a_missing_file_is_absent_and_still_not_allowed_when_not_required() -> None:
    """The ONLY non-deny outcome. A caller that ignores `mode` still fails closed."""
    verdict = policy_verdict(MISSING_PATH, required=False)
    assert verdict.mode == "absent"
    assert verdict.allowed is False
    assert verdict.is_policy_absent is True


def test_valid_policy_yields_the_declared_permissions() -> None:
    verdict = policy_verdict(FIXTURES / "valid.yaml")
    assert verdict.allowed is True
    permissions = verdict.permissions
    assert permissions is not None
    assert permissions.allowed_write_paths == ("src/**",)
    assert permissions.allowed_commands == ("pytest",)
    assert permissions.max_cost_usd == 5.0
    # `deny` wins over `allow`, so `run_commands` is not in the effective set.
    assert permissions.allowed_tools == ("read_files", "editor")
    assert "run_commands" in permissions.denied_tools
    assert "run_commands" not in permissions.effective_tools()


def test_the_digest_is_stable_across_formatting() -> None:
    """A digest that changed with key order/indentation could not pin provenance."""
    first = FIXTURES / "valid.yaml"
    copy = FIXTURES / "_tmp_reformatted.yaml"
    document = yaml.safe_load(first.read_text(encoding="utf-8"))
    # Same document, different key order and flow/block style.
    copy.write_text(
        yaml.safe_dump(dict(reversed(list(document.items()))), sort_keys=True),
        encoding="utf-8",
    )
    try:
        assert policy_verdict(first).policy_digest == policy_verdict(copy).policy_digest
    finally:
        copy.unlink()


def test_the_digest_actually_changes_when_permissions_change() -> None:
    """Guards the test above: an always-equal digest would pass it trivially."""
    original = FIXTURES / "_tmp_original.yaml"
    changed = FIXTURES / "_tmp_changed.yaml"
    original.write_text(
        "version: 1\ntools:\n  allow: [read_files]\nwrite_scope:\n  paths: ['src/**']\n",
        encoding="utf-8",
    )
    changed.write_text(
        "version: 1\ntools:\n  allow: [read_files, editor]\nwrite_scope:\n  paths: ['src/**']\n",
        encoding="utf-8",
    )
    try:
        assert policy_verdict(original).policy_digest != policy_verdict(changed).policy_digest
    finally:
        original.unlink()
        changed.unlink()


def test_policy_path_is_the_documented_location() -> None:
    assert policy_path_for("/repo") == Path("/repo/.codna/policy.yaml")


# ── the specific field-level refusals the schema documents ────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "code"),
    [
        pytest.param("version: 1\n", "policy_denies_all", id="version-only"),
        pytest.param("version: 1\nunexpected: 1\n", "policy_unknown_key", id="unknown-top"),
        pytest.param("version: 2\ntools:\n  allow: [read_files]\n", "policy_unsupported_version", id="version"),
        pytest.param("version: 1\nwrite_scope:\n  paths: ['/etc/**']\n", "policy_invalid_value", id="abs-path"),
        pytest.param("version: 1\nwrite_scope:\n  paths: ['../outside/**']\n", "policy_invalid_value", id="dotdot"),
        pytest.param("version: 1\ncommand_allow_list:\n  commands: ['/usr/bin/curl']\n", "policy_invalid_value", id="cmd-path"),
        pytest.param("version: 1\ntools:\n  deny: [editor]\n", "policy_missing_field", id="tools-no-allow"),
        pytest.param("version: 1\nbudget: {}\n", "policy_missing_field", id="empty-budget"),
        pytest.param("version: 1\ntools:\n  allow: [Editor]\n", "policy_invalid_value", id="tool-case"),
        pytest.param("version: 1\ntools:\n  allow: [read_files, 7]\n", "policy_invalid_type", id="tool-int"),
        pytest.param("version: 1\nbudget:\n  max_tokens: 1.5\n", "policy_invalid_value", id="token-fraction"),
        pytest.param("version: 1\nbudget:\n  max_cost_usd: -1\n", "policy_invalid_value", id="negative"),
        pytest.param("version: 1\nbudget:\n  max_cost_usd: true\n", "policy_invalid_type", id="bool-cap"),
    ],
)
def test_field_level_refusals(tmp_path: Path, body: str, code: str) -> None:
    target = tmp_path / "policy.yaml"
    target.write_text(body, encoding="utf-8")
    verdict = policy_verdict(target)
    assert verdict.allowed is False
    assert verdict.code == code, verdict.to_dict()


def test_the_error_type_carries_a_code_and_a_json_path(tmp_path: Path) -> None:
    target = tmp_path / "policy.yaml"
    target.write_text("version: 1\ntools:\n  allow: read_files\n", encoding="utf-8")
    with pytest.raises(PolicyDocumentError) as caught:
        load_policy_document(target)
    assert caught.value.code == "policy_invalid_type"
    assert caught.value.path == "tools.allow"
    assert caught.value.as_dict()["error"]["details"]["at"] == "tools.allow"


def test_non_utf8_document_denies(tmp_path: Path) -> None:
    target = tmp_path / "policy.yaml"
    target.write_bytes(b"version: 1\ntools:\n  allow: [\xff\xfe]\n")
    assert policy_verdict(target).code == "policy_unparseable"


def test_deep_nesting_is_refused(tmp_path: Path) -> None:
    """A 40-level document must be refused by the depth bound, not walked to the stack limit.

    The schema is three levels deep, so anything past `MAX_DOCUMENT_DEPTH` is malformed. Written as
    block YAML so the parser accepts it and the DEPTH check is what refuses it (a flow-style bomb is
    rejected by PyYAML's own scanner first, which is also a refusal but proves nothing about this
    module's guard).
    """
    lines = ["version: 1"]
    for level in range(40):
        lines.append("  " * level + f"n{level}:")
    lines.append("  " * 40 + "1")
    target = tmp_path / "policy.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    verdict = policy_verdict(target)
    assert verdict.allowed is False
    assert verdict.code == "policy_invalid_shape"


def test_a_cyclic_document_is_refused(tmp_path: Path) -> None:
    """`deny: *t` points the list at its own parent. PyYAML refuses to construct it, which is the
    correct outcome; the module's own cycle guard covers a cycle that a future loader might build."""
    target = tmp_path / "policy.yaml"
    target.write_text("version: 1\ntools: &t\n  allow: [read_files]\n  deny: *t\n", encoding="utf-8")
    verdict = policy_verdict(target)
    assert verdict.allowed is False
    assert verdict.code == "policy_unparseable"


# ── the CLI the sidecar shells out to ────────────────────────────────────────────────────────


def test_cli_emits_one_json_object_and_exits_zero_for_a_denial(tmp_path: Path, capsys) -> None:
    """A DENIAL is a successful evaluation, so the caller can report the policy's own code."""
    target = tmp_path / "policy.yaml"
    target.write_text("version: 1\nnope: 1\n", encoding="utf-8")
    assert main(["verify", "--file", str(target)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allowed"] is False
    assert payload["code"] == "policy_unknown_key"


def test_cli_exits_two_without_a_verdict_on_bad_usage(capsys) -> None:
    """No verdict = the caller cannot distinguish deny from failure, so it must deny."""
    assert main(["verify", "--file"]) == 2
    assert main(["frobnicate"]) == 2
    capsys.readouterr()


def test_cli_is_runnable_as_a_subprocess() -> None:
    """The sidecar runs `python3 -m codna.policy_cli`, not an in-process call."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "codna.policy_cli",
            "verify",
            "--file",
            str(FIXTURES / "valid.yaml"),
            "--required",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["allowed"] is True
    assert payload["mode"] == "policy"


def test_cli_optional_flag_reports_absence(tmp_path: Path, capsys) -> None:
    assert main(["verify", "--file", str(tmp_path / "nope.yaml"), "--optional"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "absent"
    assert payload["allowed"] is False
