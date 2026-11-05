"""The `durable` extra, its pins, and `require_durable`'s fail-fast contract.

Two audiences. Packaging: the extra must exist, must not leak into base dependencies, and
`constraints-durable.txt` must agree with what `require_durable` expects — a release that pins one
version and ships code expecting another is the failure mode exact pins are supposed to prevent, so
it is a test rather than a comment.

Behaviour: the base suite runs WITHOUT the extra, so the missing-extra path is exercised for real
here, and the installed path is exercised whenever the extra is present. Both branches have to be
proven, because "fails fast with a hint" and "works" are different claims.
"""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from codna import durable
from codna.durable import DURABLE_PINS, DURABLE_RANGES, INSTALL_HINT, DurableUnavailableError, require_durable

CLI_DIR = Path(__file__).resolve().parents[2]
PYPROJECT = CLI_DIR / "pyproject.toml"
CONSTRAINTS = CLI_DIR / "constraints-durable.txt"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


# -- packaging --------------------------------------------------------------------------


def test_the_durable_extra_is_declared_with_the_documented_ranges():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "durable" in extras
    declared = extras["durable"]
    assert {item.distribution for item in DURABLE_PINS} == set(DURABLE_RANGES)
    for name, range_text in DURABLE_RANGES.items():
        # `temporalio` carries an extra (`[langgraph]`): the plain wheel has no contrib plugin, so the
        # requirement is asserted by name + range rather than by exact string.
        matches = [item for item in declared if item.startswith(name) and range_text in item]
        assert matches, f"{name}{range_text} not declared in the extra: {declared}"
    assert any(item.startswith("temporalio[langgraph]") for item in declared)
    assert "pyjwt>=2.8,<3" in declared, "the durable Run API's OIDC verifier requires PyJWT"


def test_no_durable_dependency_leaks_into_the_base_install():
    dependencies = " ".join(_pyproject()["project"]["dependencies"])
    assert "temporalio" not in dependencies
    assert "langgraph" not in dependencies
    assert "langchain" not in dependencies
    assert "pyjwt" not in dependencies


def test_constraints_file_pins_exactly_what_require_durable_expects():
    """The single source of truth check: code pins == constraints pins."""
    text = CONSTRAINTS.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    pinned = {}
    for line in lines:
        name, _, version = line.partition("==")
        assert version, f"constraint {line!r} is not an exact pin"
        pinned[name.strip()] = version.strip()

    for requirement in DURABLE_PINS:
        assert requirement.distribution in pinned, f"{requirement.distribution} missing from constraints"
        assert pinned[requirement.distribution] == requirement.pin, (
            f"{requirement.distribution}: constraints pin {pinned[requirement.distribution]!r} "
            f"!= code pin {requirement.pin!r}"
        )
    assert pinned.get("pyjwt") == "2.10.1", "the OIDC verifier must use the tested PyJWT pin"


def test_the_durable_package_is_declared_for_packaging():
    packages = _pyproject()["tool"]["setuptools"]["packages"]
    assert "codna.durable" in packages


def test_the_install_hint_is_the_extra_install_command():
    assert INSTALL_HINT == "pip install 'codna[durable]'"


# -- require_durable: the missing branch -------------------------------------------------


def test_missing_extra_raises_a_structured_error_with_a_hint():
    """The branch that runs in the BASE suite: the extra is genuinely absent here."""
    if importlib.util.find_spec("temporalio") is not None:
        pytest.skip("temporalio IS installed in this interpreter; the branch is covered by the simulation below")
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    error = excinfo.value
    assert error.code == "durable_extra_missing"
    assert error.hint == INSTALL_HINT
    assert "temporalio" in error.details["missing"]
    # The structured body is what the CLI prints and the Run API returns.
    body = error.as_dict()["error"]
    assert body["code"] == "durable_extra_missing"
    assert body["hint"] == INSTALL_HINT
    # The message must be actionable on its own: a user reading only the message knows what to do.
    assert INSTALL_HINT in str(error)


def test_missing_extra_is_simulated_even_when_it_is_installed(monkeypatch):
    """Deterministic coverage of the missing branch regardless of the interpreter."""
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: ["temporalio"])
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    assert excinfo.value.code == "durable_extra_missing"
    assert excinfo.value.details["missing"] == ["temporalio"]


def test_a_missing_plugin_is_a_distinct_code(monkeypatch):
    """`temporalio` without `temporalio[langgraph]` has its own remedy, so its own code."""
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _fail_plugin_import)
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    assert excinfo.value.code == "durable_plugin_missing"
    assert "temporalio[langgraph]" in str(excinfo.value)
    assert "temporalio[langgraph]" in excinfo.value.details["install"]


def _stub_import(name: str, *args: object, **kwargs: object):
    """Stand in for a fully-installed stack, so only the version gate is under test."""
    import types

    return types.ModuleType(name)


def _fail_plugin_import(name: str, *args: object, **kwargs: object):
    """Stand in for a `temporalio` install without the langgraph contrib."""
    if name == durable.PLUGIN_MODULE:
        raise ImportError("no langgraph contrib")
    return importlib.import_module(name)


def test_an_out_of_window_version_fails_fast_with_the_pin_in_the_hint(monkeypatch):
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _stub_import)
    monkeypatch.setattr(durable, "_installed_version", lambda name: "0.99.0")
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    assert excinfo.value.code == "durable_version_unsupported"
    assert excinfo.value.details["installed"] == "0.99.0"
    assert "==1.34.0" in excinfo.value.hint or "==1.2.17" in excinfo.value.hint


def test_a_newer_major_is_refused_so_a_preview_bump_cannot_slip_in(monkeypatch):
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _stub_import)

    def versions(name: str) -> str:
        return {"temporalio": "2.0.0", "langgraph": "1.2.17", "langchain-core": "1.7.0",
                "langgraph-checkpoint": "4.3.0"}[name]

    monkeypatch.setattr(durable, "_installed_version", versions)
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    assert excinfo.value.code == "durable_version_unsupported"
    assert excinfo.value.details["distribution"] == "temporalio"


# -- version parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.34.0", (1, 34, 0)),
        ("1.2.17", (1, 2, 17)),
        ("1.34.0b2", (1, 34, 0)),
        ("1.0.0+local", (1, 0, 0)),
        ("2", (2,)),
    ],
)
def test_version_parsing_handles_pre_releases_and_local_suffixes(raw, expected):
    assert durable._parse_version(raw) == expected


def test_version_parsing_rejects_nonsense():
    with pytest.raises(ValueError):
        durable._parse_version("not-a-version")


# -- require_durable: the installed branch ----------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("temporalio") is None, reason="the `durable` extra is not installed")
def test_require_durable_returns_the_stack_when_the_extra_is_installed():
    if importlib.util.find_spec("temporalio") is None:
        pytest.skip("the `durable` extra is not installed in this interpreter")
    stack = require_durable()
    assert stack.temporalio.__name__ == "temporalio"
    assert stack.langgraph.__name__ == "langgraph"
    assert stack.langchain_core.__name__ == "langchain_core"
    assert stack.langgraph_plugin is not None
    for requirement in DURABLE_PINS:
        assert requirement.distribution in stack.versions
        parsed = durable._parse_version(stack.versions[requirement.distribution])
        assert requirement.minimum <= parsed < requirement.maximum


@pytest.mark.skipif(importlib.util.find_spec("temporalio") is None, reason="the `durable` extra is not installed")
def test_require_durable_works_on_a_BARE_interpreter(tmp_path):
    """Regression: `require_durable` called `importlib.util.find_spec` without importing `importlib.util`.

    It worked in this suite only because something else in the process had already imported
    `importlib.util` (pytest does, among others). In the HOSTED WORKER image — a bare interpreter whose
    first codna import is `codna.durable` — it raised `AttributeError: module 'importlib' has no
    attribute 'util'`, which is exactly what the image's health check and `codna worker` do. Found by
    running the built image, not by the unit tests; this pins it.

    `-S` (no site) is not needed: the PROPERTY is that `codna.durable` alone leaves `importlib.util`
    available, so the subprocess imports nothing before it.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        # Start from a state where `importlib.util` has NOT been imported by anything else: drop it if
        # the interpreter's startup put it there, so the assertion below cannot pass vacuously.
        import importlib
        for name in [n for n in list(sys.modules) if n == "importlib.util"]:
            del sys.modules[name]
        importlib.__dict__.pop("util", None)
        assert not hasattr(importlib, "util"), "the harness could not clear importlib.util"

        from codna.durable import require_durable
        require_durable()
        print("ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert completed.returncode == 0, (
        "`require_durable()` must work on an interpreter where nothing imported importlib.util "
        f"first — that is the container's state:\n{completed.stderr}"
    )
    assert completed.stdout.strip() == "ok"


def test_importing_the_durable_package_does_not_import_temporalio():
    """`codna --version` and the base test suite must not pay for the extra."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import codna.durable
        print("temporalio" in sys.modules, "langgraph" in sys.modules)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=str(CLI_DIR), capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False False"


def test_an_out_of_window_checkpoint_version_reports_its_supported_range(monkeypatch):
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _stub_import)

    def versions(name: str) -> str:
        return {
            "temporalio": "1.34.0",
            "langgraph": "1.2.17",
            "langchain-core": "1.7.0",
            "langgraph-checkpoint": "5.0.0",
        }[name]

    monkeypatch.setattr(durable, "_installed_version", versions)
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    assert excinfo.value.code == "durable_version_unsupported"
    assert excinfo.value.details["distribution"] == "langgraph-checkpoint"
    assert excinfo.value.details["range"] == ">=4,<5"


def test_base_cli_parser_and_version_do_not_need_optional_durable_imports():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys

        class BlockOptionalDurable:
            def find_spec(self, fullname, path=None, target=None):
                blocked = {"temporalio", "langgraph", "langchain_core", "jwt"}
                if fullname.partition(".")[0] in blocked:
                    raise ModuleNotFoundError(f"blocked optional durable dependency: {fullname}")

        sys.meta_path.insert(0, BlockOptionalDurable())
        from codna.cli import build_parser, main

        parser = build_parser()
        subcommands = next(action for action in parser._actions if getattr(action, "choices", None))
        assert {"durable", "run", "worker"} <= set(subcommands.choices)
        try:
            main(["--version"])
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("--version must exit")
        print("base-cli-ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(CLI_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[-1] == "base-cli-ok"
