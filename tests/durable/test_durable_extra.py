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
from codna.durable import (
    DURABLE_PINS,
    DURABLE_RANGES,
    INSTALL_HINT,
    PLUGIN_DISTRIBUTION,
    PLUGIN_PIN,
    PLUGIN_RANGE,
    DurableUnavailableError,
    require_durable,
)

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
    # PyJWT is deliberately NOT in `DURABLE_PINS` (it has no module in `DurableStack`), so this
    # literal cannot be derived from the code the way the four stack pins are. It is the REVIEWED
    # value: a pyjwt bump has to be restated here, which is what makes it a decision rather than a
    # consequence of the constraints file — otherwise the assertion would only compare that file
    # with itself.
    assert pinned.get("pyjwt") == "2.15.1", "the OIDC verifier must use the tested PyJWT pin"
    # The Temporal LangGraph plugin, likewise reviewed rather than derived — and for the extra reason
    # that it cannot join `DURABLE_PINS` without narrowing that gate's `temporalio>=1.34` window (1.34
    # has no such distribution; it vendored the plugin). Since `temporalio` 1.35 this is a 0.x package
    # that owns the node→activity mapping, so an unpinned resolve is the replay-break this file exists
    # to prevent. Asserted here so removing the pin is a test failure, not a silent drift.
    assert pinned.get("temporalio-langgraph") == "0.1.0", (
        "the Temporal LangGraph plugin must stay pinned: it owns the node→activity mapping, and 0.x "
        "minor releases are allowed to change it"
    )


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
    # Derived from the code's own pin, not a literal: the hint must name the version this build
    # expects, so a bump cannot leave the assertion passing on a stale string.
    assert f"=={DURABLE_PINS[0].pin}" in excinfo.value.hint


def test_the_plugin_distribution_is_enforced_at_runtime_when_it_is_present(monkeypatch):
    """The plugin owns the node→activity mapping, so its version is a RUNTIME gate, not a pin.

    It is not in `DURABLE_PINS` (see `PLUGIN_DISTRIBUTION`: 1.34 vendored the plugin, so requiring
    the distribution there would refuse a supported install). That makes this check the only thing
    standing between a stray `temporalio-langgraph` and a silent replay break, so each branch of it
    is asserted rather than assumed.
    """
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _stub_import)

    def versions(name: str) -> str | None:
        installed = {requirement.distribution: requirement.pin for requirement in DURABLE_PINS}
        installed[PLUGIN_DISTRIBUTION] = "0.2.0"  # outside `>=0.1,<0.2`
        # `.get`, not `[name]`: the real `_installed_version` returns None for an uninstalled
        # distribution, and a KeyError here would surface as an unexpected exception instead of
        # the `DurableUnavailableError` this test is about.
        return installed.get(name)

    monkeypatch.setattr(durable, "_installed_version", versions)
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    assert excinfo.value.code == "durable_version_unsupported"
    assert excinfo.value.details["distribution"] == PLUGIN_DISTRIBUTION
    assert excinfo.value.details["installed"] == "0.2.0"
    assert excinfo.value.details["range"] == PLUGIN_RANGE
    assert f"=={PLUGIN_PIN}" in excinfo.value.hint


def test_the_plugin_absence_is_not_an_error_so_1_34_keeps_working(monkeypatch):
    """A 1.34 install has no `temporalio-langgraph`; the `temporalio` pin is what guards it there.

    Refusing an absent plugin would break the window the extra still declares (`temporalio>=1.34`),
    so absence must stay legal — this is the branch that keeps that true.
    """
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _stub_import)

    def versions(name: str) -> str:
        installed = {requirement.distribution: requirement.pin for requirement in DURABLE_PINS}
        return installed.get(name, "")  # plugin -> "", i.e. not installed

    monkeypatch.setattr(durable, "_installed_version", versions)
    stack = require_durable()
    assert PLUGIN_DISTRIBUTION not in stack.versions


def test_an_installed_plugin_is_reported_in_the_stack_versions(monkeypatch):
    """A caller can see which plugin build it got, which is what makes a support report actionable."""
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _stub_import)

    def versions(name: str) -> str | None:
        installed = {requirement.distribution: requirement.pin for requirement in DURABLE_PINS}
        installed[PLUGIN_DISTRIBUTION] = PLUGIN_PIN
        return installed.get(name)

    monkeypatch.setattr(durable, "_installed_version", versions)
    stack = require_durable()
    assert stack.versions[PLUGIN_DISTRIBUTION] == PLUGIN_PIN


@pytest.mark.parametrize(
    "raw,accepted",
    [
        ("0.1.0", True),
        ("0.1.9", True),
        ("0.1.0b1", True),
        ("0.0.9", False),
        ("0.2.0", False),
        ("0.2.0a1", False),
        ("1.0.0", False),
    ],
)
def test_the_plugin_window_is_the_declared_range(raw, accepted):
    assert durable._plugin_accepts(durable._parse_version(raw)) is accepted


def test_a_newer_major_is_refused_so_a_preview_bump_cannot_slip_in(monkeypatch):
    monkeypatch.setattr(durable, "_missing_modules", lambda requirements: [])
    monkeypatch.setattr(durable.importlib, "import_module", _stub_import)

    def versions(name: str) -> str:
        # Every OTHER pin is the code's own, so only `temporalio`'s major is under test: a literal
        # here would let a bump leave this test asserting yesterday's stack.
        installed = {requirement.distribution: requirement.pin for requirement in DURABLE_PINS}
        installed["temporalio"] = "2.0.0"
        return installed[name]

    monkeypatch.setattr(durable, "_installed_version", versions)
    with pytest.raises(DurableUnavailableError) as excinfo:
        require_durable()
    assert excinfo.value.code == "durable_version_unsupported"
    assert excinfo.value.details["distribution"] == "temporalio"


# -- version parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.34.0", ((1, 34, 0), False)),
        ("1.2.17", ((1, 2, 17), False)),
        ("1.34.0b2", ((1, 34, 0), True)),
        ("1.34.dev0", ((1, 34), True)),
        ("1.0.0+local", ((1, 0, 0), False)),
        ("2", ((2,), False)),
    ],
)
def test_version_parsing_handles_pre_releases_and_local_suffixes(raw, expected):
    assert durable._parse_version(raw) == expected


@pytest.mark.parametrize("raw", ["1.34dev0", "1.34.0b1", "1.34.0rc1"])
def test_a_prerelease_at_the_inclusive_minimum_release_is_allowed(raw):
    assert DURABLE_PINS[0].accepts(durable._parse_version(raw))


def test_a_prerelease_at_the_excluded_upper_major_is_rejected():
    assert not DURABLE_PINS[0].accepts(durable._parse_version("2.0.0a1"))


def test_a_prerelease_before_the_excluded_upper_major_is_allowed():
    assert DURABLE_PINS[0].accepts(durable._parse_version("1.99.0a1"))


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
        assert requirement.accepts(parsed)


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
        # As above, in reverse: only the CHECKPOINT pin is out of window here.
        installed = {requirement.distribution: requirement.pin for requirement in DURABLE_PINS}
        installed["langgraph-checkpoint"] = "5.0.0"
        return installed[name]

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
