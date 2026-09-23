from __future__ import annotations


import pytest

from codna import testrun


_JUNIT = """<?xml version="1.0"?>
<testsuite tests="3" failures="1" errors="1">
  <testcase classname="pkg.test_mod" name="test_ok"/>
  <testcase classname="pkg.test_mod" name="test_bad"><failure>boom</failure></testcase>
  <testcase classname="pkg.test_mod" name="test_err"><error>kaboom</error></testcase>
</testsuite>
"""


def test_parse_junit_failures(tmp_path):
    p = tmp_path / "report.xml"
    p.write_text(_JUNIT, encoding="utf-8")
    ids = testrun.parse_junit_failures(str(p))
    assert ids == ["pkg.test_mod::test_bad", "pkg.test_mod::test_err"]


# A JUnit report is produced by an UNTRUSTED repo's test run: an entity declaration (XXE, or a
# billion-laughs expansion bomb) must be rejected, never resolved/expanded.
_JUNIT_XXE = """<?xml version="1.0"?>
<!DOCTYPE testsuite [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
  <!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
]>
<testsuite tests="1" failures="1">
  <testcase classname="pkg.test_mod" name="test_bad"><failure>&xxe;&lol2;</failure></testcase>
</testsuite>
"""


def test_parse_junit_failures_rejects_entity_declarations(tmp_path):
    from defusedxml import DefusedXmlException

    p = tmp_path / "report.xml"
    p.write_text(_JUNIT_XXE, encoding="utf-8")
    with pytest.raises(DefusedXmlException) as excinfo:
        testrun.parse_junit_failures(str(p))
    assert isinstance(excinfo.value, ValueError)
    # ...and `codna fix --from-junit` turns that into a clean bad-input error, not a traceback.
    from codna import fix_inputs

    with pytest.raises(fix_inputs.FixInputError, match="could not read JUnit report"):
        fix_inputs.from_junit(str(p))
    # The collection-error probe on the same payload is a plain "no" rather than an expansion.
    assert testrun.junit_has_collection_errors(str(p)) is False


# ── monorepo-aware detection: pytest config one level down (codna's OWN repo shape) ───────────
def test_detect_pytest_finds_config_in_a_depth1_subdirectory(tmp_path):
    (tmp_path / "cli" / "tests").mkdir(parents=True)
    assert testrun._detect_pytest(str(tmp_path)) == str(tmp_path / "cli")


def test_detect_pytest_prefers_repo_root_over_a_subdirectory(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "cli" / "tests").mkdir(parents=True)
    assert testrun._detect_pytest(str(tmp_path)) == str(tmp_path)


def test_detect_pytest_raises_when_multiple_subdirectories_are_ambiguous(tmp_path):
    """codna's OWN repo shape: bench/, cli/, and python/ are all independently valid pytest
    roots with nothing at the repo root -- must not silently pick one (e.g. alphabetically)."""
    (tmp_path / "bench" / "tests").mkdir(parents=True)
    (tmp_path / "cli" / "tests").mkdir(parents=True)
    with pytest.raises(RuntimeError) as exc:
        testrun._detect_pytest(str(tmp_path))
    assert "--test-cmd" in str(exc.value)
    assert "bench" in str(exc.value) and "cli" in str(exc.value)


def test_detect_pytest_none_when_neither_root_nor_any_subdirectory_has_config(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "node_modules" / "tests").mkdir(parents=True)  # skip-listed, must not count
    assert testrun._detect_pytest(str(tmp_path)) is None


def test_discover_runs_from_the_subdirectory_that_actually_has_pytest_config(tmp_path, monkeypatch):
    """Regression: codna's own repo has cli/pyproject.toml + cli/tests/, nothing at the root --
    `codna fix --tests` on codna's own repo failed outright with "could not detect a test
    runner" until _detect_pytest looked one level down. Detecting it is not enough on its own:
    pytest must also actually RUN from that subdirectory (not the repo root) to pick up its
    pyproject.toml's own [tool.pytest.ini_options] (testpaths, markers, addopts, ...)."""
    (tmp_path / "cli" / "tests").mkdir(parents=True)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY", raising=False)
    captured = {}

    class _FakeResult:
        timed_out = False
        returncode = 0
        stdout = ""
        stderr = ""

    class _FakeSandbox:
        def __init__(self, **kw):
            pass

        def run(self, argv, *, cwd, accepted_exit_codes=(0,)):
            captured["cwd"] = cwd
            return _FakeResult()

    monkeypatch.setattr(testrun, "Sandbox", _FakeSandbox)
    testrun.discover_failing_tests(str(tmp_path))
    assert captured["cwd"] == str(tmp_path / "cli")  # NOT the repo root


def test_discover_raises_without_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(testrun, "_detect_pytest", lambda d: False)
    with pytest.raises(RuntimeError) as exc:
        testrun.discover_failing_tests(str(tmp_path))
    assert "--test-cmd" in str(exc.value)


def test_discover_runs_sandboxed_and_parses(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()  # makes _detect_pytest True
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())  # pretend pytest present
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY", raising=False)  # isolate from config-apply leakage

    captured = {}

    class _FakeResult:
        timed_out = False
        returncode = 1
        stdout = ""
        stderr = ""

    class _FakeSandbox:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self, argv, *, cwd, accepted_exit_codes=(0,)):
            captured["argv"] = argv
            captured["cwd"] = cwd
            captured["accepted"] = accepted_exit_codes
            # write the junit the sandbox "run" would produce
            junit = argv[-1].split("=", 1)[1]
            with open(junit, "w", encoding="utf-8") as fh:
                fh.write(_JUNIT)
            return _FakeResult()

    monkeypatch.setattr(testrun, "Sandbox", _FakeSandbox)
    issue, failing = testrun.discover_failing_tests(str(tmp_path))

    assert failing == ["pkg.test_mod::test_bad", "pkg.test_mod::test_err"]
    assert "2 failing test(s)" in issue
    # network denied + scrubbed env (no write token)
    assert captured["network"] == "deny"
    assert "GITHUB_TOKEN" not in captured["env"]
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["accepted"] == (0, 1)  # pytest exits 1 on failures — accepted, not an error


def test_custom_test_cmd_uses_sh(tmp_path, monkeypatch):
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY", raising=False)  # isolate from config-apply leakage
    captured = {}

    class _FakeResult:
        timed_out = False
        returncode = 0
        stdout = ""
        stderr = ""

    class _FakeSandbox:
        def __init__(self, **kw):
            pass

        def run(self, argv, *, cwd, accepted_exit_codes=(0,)):
            captured["argv"] = argv
            return _FakeResult()

    class _CapturingSandbox:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self, argv, *, cwd, accepted_exit_codes=(0,)):
            captured["argv"] = argv
            return _FakeResult()

    monkeypatch.setattr(testrun, "Sandbox", _CapturingSandbox)
    testrun.discover_failing_tests(str(tmp_path), test_cmd="make test")
    assert captured["argv"][:2] == ["/bin/sh", "-c"]
    assert captured["argv"][2] == "make test"
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_scrub_secrets_removes_provider_and_cloud_keys():
    env = {
        "PATH": "/usr/bin", "HOME": "/h", "LANG": "C",
        "ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-o", "CODNA_API_KEY": "ck",
        "AWS_SECRET_ACCESS_KEY": "aws", "GEMINI_API_KEY": "g", "MY_SERVICE_TOKEN": "t",
        "DB_PASSWORD": "p", "TELYS_TOKEN": "tt",
    }
    out = testrun._scrub_secrets(env)
    assert out == {"PATH": "/usr/bin", "HOME": "/h", "LANG": "C"}  # only non-secret env survives


def test_discover_wires_network_backend_and_scrubs(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(testrun, "default_network_backend", lambda: "REAL_BACKEND")
    captured = {}

    class _R:
        timed_out = False
        returncode = 0
        stdout = ""
        stderr = ""

    class _FakeSandbox:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self, argv, *, cwd, accepted_exit_codes=(0,)):
            return _R()

    monkeypatch.setattr(testrun, "Sandbox", _FakeSandbox)
    testrun.discover_failing_tests(str(tmp_path))
    assert captured["network_backend"] == "REAL_BACKEND"   # egress actually enforced, not just recorded
    assert "ANTHROPIC_API_KEY" not in captured["env"]      # provider key scrubbed


def test_discover_fail_closed_when_egress_required_but_no_backend(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(testrun, "default_network_backend", lambda: None)
    monkeypatch.setenv("CODNA_REQUIRE_EGRESS_DENY", "1")
    import pytest
    with pytest.raises(RuntimeError) as exc:
        testrun.discover_failing_tests(str(tmp_path))
    assert "egress" in str(exc.value).lower()



def test_junit_collection_errors_are_detected_and_plain_failures_are_not(tmp_path):
    from codna.testrun import junit_has_collection_errors

    bad = tmp_path / "collection.xml"
    bad.write_text('<testsuites><testsuite><testcase classname="" name="tests/test_x.py">'
                   '<error message="collection failure">ModuleNotFoundError: No module named httpx</error>'
                   '</testcase></testsuite></testsuites>')
    good = tmp_path / "failure.xml"
    good.write_text('<testsuites><testsuite><testcase classname="tests.test_x" name="test_y">'
                    '<failure message="assert 1 == 2">boom</failure></testcase></testsuite></testsuites>')
    assert junit_has_collection_errors(str(bad)) is True
    assert junit_has_collection_errors(str(good)) is False
    assert junit_has_collection_errors(str(tmp_path / "missing.xml")) is False


# ── per-repo test command: codna.yaml fix.test_command > pixi/uv detection > pytest ───────────
class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.timed_out = False
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _capture_sandbox(monkeypatch, *, returncode=0, junit_text=None, stderr=""):
    """A fake Sandbox that records argv/env/cwd and, when asked, writes a JUnit report where the
    run would (``$CODNA_JUNIT`` -- the same path the pytest default passes as --junitxml)."""
    captured = {}

    class _Sandbox:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self, argv, *, cwd, accepted_exit_codes=(0,)):
            captured["argv"] = argv
            captured["cwd"] = cwd
            if junit_text is not None:
                with open(captured["env"]["CODNA_JUNIT"], "w", encoding="utf-8") as fh:
                    fh.write(junit_text)
            return _Result(returncode, stderr=stderr)

    monkeypatch.setattr(testrun, "Sandbox", _Sandbox)
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY", raising=False)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    return captured


def test_detect_pixi_test_task_in_pixi_toml(tmp_path):
    (tmp_path / "pixi.toml").write_text(
        '[workspace]\nname = "k"\n[tasks]\ntest = { cmd = "bash scripts/test_all.sh", depends-on = ["build"] }\n',
        encoding="utf-8")
    assert testrun.detect_test_command(str(tmp_path)) == ("pixi run test", "pixi.toml")


def test_detect_pixi_string_task_and_pyproject_tool_pixi(tmp_path):
    (tmp_path / "pixi.toml").write_text('[tasks]\ntest = "pytest -q"\n', encoding="utf-8")
    assert testrun.detect_test_command(str(tmp_path)) == ("pixi run test", "pixi.toml")
    (tmp_path / "pixi.toml").unlink()
    (tmp_path / "pyproject.toml").write_text('[tool.pixi.tasks]\ntest = "pytest -q"\n', encoding="utf-8")
    assert testrun.detect_test_command(str(tmp_path)) == ("pixi run test", "pyproject.toml")


def test_detect_pixi_without_a_test_task_is_none(tmp_path):
    (tmp_path / "pixi.toml").write_text('[tasks]\nbuild = "make"\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()
    assert testrun.detect_test_command(str(tmp_path)) is None  # falls through to the pytest default


def test_detect_ignores_a_pyproject_without_tool_pixi(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n', encoding="utf-8")
    assert testrun.detect_test_command(str(tmp_path)) is None


def test_detect_uv_needs_a_lock_that_pins_pytest_next_to_pytest_config(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "requests"\nversion = "2.0"\n', encoding="utf-8")
    assert testrun.detect_test_command(str(tmp_path)) is None
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "pytest"\nversion = "8.0"\n', encoding="utf-8")
    assert testrun.detect_test_command(str(tmp_path)) == ("uv run pytest", "uv.lock")


def test_repo_test_command_reads_the_fix_block_from_the_repos_codna_yaml(tmp_path):
    (tmp_path / "codna.yaml").write_text("fix:\n  test_command: pixi run test\n", encoding="utf-8")
    assert testrun.repo_test_command(str(tmp_path)) == "pixi run test"
    (tmp_path / "codna.yaml").unlink()
    (tmp_path / ".codna.yaml").write_text("fix:\n  test_command: make test\n", encoding="utf-8")
    assert testrun.repo_test_command(str(tmp_path)) == "make test"


def test_repo_test_command_none_when_absent_or_without_a_fix_block(tmp_path):
    assert testrun.repo_test_command(str(tmp_path)) is None
    (tmp_path / "codna.yaml").write_text("privacy:\n  redact_secrets: true\n", encoding="utf-8")
    assert testrun.repo_test_command(str(tmp_path)) is None


@pytest.mark.parametrize("text", ["fix:\n  test_command: 3\n", "fix:\n  test_command: ''\n", "fix: pixi run test\n"])
def test_repo_test_command_fails_closed_on_a_bad_value(tmp_path, text):
    from codna.config_file import ConfigError

    (tmp_path / "codna.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        testrun.repo_test_command(str(tmp_path))


def test_resolve_precedence_flag_then_config_then_detection_then_pytest(tmp_path):
    (tmp_path / "pixi.toml").write_text('[tasks]\ntest = "pytest"\n', encoding="utf-8")
    (tmp_path / "codna.yaml").write_text("fix:\n  test_command: make check\n", encoding="utf-8")
    assert testrun.resolve_test_command(str(tmp_path), "just test") == ("just test", "--test-cmd", True)
    assert testrun.resolve_test_command(str(tmp_path)) == ("make check", "codna.yaml fix.test_command", True)
    (tmp_path / "codna.yaml").unlink()
    assert testrun.resolve_test_command(str(tmp_path)) == ("pixi run test", "pixi.toml", False)
    (tmp_path / "pixi.toml").unlink()
    assert testrun.resolve_test_command(str(tmp_path)) is None


def test_discover_runs_the_configured_command_and_asks_pytest_for_the_junit(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()  # pytest config exists too -- the configured command still wins
    (tmp_path / "codna.yaml").write_text("fix:\n  test_command: pixi run test\n", encoding="utf-8")
    captured = _capture_sandbox(monkeypatch, returncode=1, junit_text=_JUNIT)
    issue, failing = testrun.discover_failing_tests(str(tmp_path))
    assert captured["argv"] == ["/bin/sh", "-c", "pixi run test"]
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["PYTEST_ADDOPTS"] == f"--junitxml={captured['env']['CODNA_JUNIT']}"
    assert failing == ["pkg.test_mod::test_bad", "pkg.test_mod::test_err"]


def test_discover_appends_to_the_repos_own_pytest_addopts(tmp_path, monkeypatch):
    captured = _capture_sandbox(monkeypatch)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p no:cacheprovider")
    testrun.discover_failing_tests(str(tmp_path), test_cmd="make test")
    assert captured["env"]["PYTEST_ADDOPTS"].startswith("-p no:cacheprovider --junitxml=")


def test_discover_detected_pixi_runs_pixi_when_it_is_installed(tmp_path, monkeypatch):
    (tmp_path / "pixi.toml").write_text('[tasks]\ntest = "pytest"\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()  # bare pytest here could never import the suite -- pixi must win
    monkeypatch.setattr(testrun.shutil, "which", lambda name: "/usr/local/bin/pixi")
    captured = _capture_sandbox(
        monkeypatch, junit_text='<testsuite><testcase classname="t" name="ok"/></testsuite>')
    assert testrun.discover_failing_tests(str(tmp_path)) == (None, [])
    assert captured["argv"] == ["/bin/sh", "-c", "pixi run test"]


def test_discover_detected_pixi_is_an_environment_gap_when_pixi_is_missing(tmp_path, monkeypatch):
    """thyn-ai/mojo-kernels#1 in the hosted image: the answer is 'this sandbox cannot run pixi' with
    the way out -- never a red check over the pull request, never bare pytest on a suite that
    cannot import."""
    (tmp_path / "pixi.toml").write_text('[tasks]\ntest = "pytest"\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(testrun.shutil, "which", lambda name: None)
    captured = _capture_sandbox(monkeypatch)
    with pytest.raises(testrun.TestEnvironmentUnavailable) as exc:
        testrun.discover_failing_tests(str(tmp_path))
    assert "argv" not in captured  # nothing ran
    assert exc.value.code == "test_environment_unavailable"
    msg = str(exc.value)
    assert "`pixi`" in msg and "--from-junit" in msg
    assert "Tell Codna how" not in msg  # the command is known; asking to configure it would not help


def test_discover_exit_127_from_a_configured_command_is_an_environment_gap(tmp_path, monkeypatch):
    (tmp_path / "codna.yaml").write_text("fix:\n  test_command: pixi run test\n", encoding="utf-8")
    _capture_sandbox(monkeypatch, returncode=127, stderr="sh: pixi: not found")
    with pytest.raises(testrun.TestEnvironmentUnavailable) as exc:
        testrun.discover_failing_tests(str(tmp_path))
    msg = str(exc.value)
    assert "exited 127" in msg and "codna.yaml fix.test_command" in msg
    assert "--from-junit" in msg and "Tell Codna how" not in msg  # already configured: no "set it" hint
    assert exc.value.details["tail"] == "sh: pixi: not found"


def test_discover_collection_errors_are_an_environment_gap_with_the_config_hint(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    collection = ('<testsuites><testsuite><testcase classname="" name="tests/test_x.py">'
                  '<error message="collection failure">ModuleNotFoundError</error></testcase></testsuite></testsuites>')
    _capture_sandbox(monkeypatch, returncode=2, junit_text=collection)
    with pytest.raises(testrun.TestEnvironmentUnavailable) as exc:
        testrun.discover_failing_tests(str(tmp_path))
    assert "could not import" in str(exc.value) and "`fix.test_command`" in str(exc.value)
    assert isinstance(exc.value, RuntimeError)  # callers that caught RuntimeError still do


def test_discover_detected_command_dying_before_pytest_is_not_failing_tests(tmp_path, monkeypatch):
    (tmp_path / "pixi.toml").write_text('[tasks]\ntest = "pytest"\n', encoding="utf-8")
    monkeypatch.setattr(testrun.shutil, "which", lambda name: "/usr/local/bin/pixi")
    _capture_sandbox(monkeypatch, returncode=1, stderr="error: failed to solve the environment (offline)")
    with pytest.raises(testrun.TestEnvironmentUnavailable) as exc:
        testrun.discover_failing_tests(str(tmp_path))
    assert "before pytest produced a report" in str(exc.value)


def test_discover_explicit_command_without_junit_still_reports_failing_tests(tmp_path, monkeypatch):
    """Regression guard: a person's own command is trusted -- non-zero means the tests failed, even
    without per-test ids (fix_run's --apply verification loop keys on this)."""
    _capture_sandbox(monkeypatch, returncode=1, stderr="FAIL: 3 tests")
    issue, failing = testrun.discover_failing_tests(str(tmp_path), test_cmd="make test")
    assert failing == [] and issue.startswith("tests are failing")


def test_discover_no_runner_anywhere_is_an_environment_gap(tmp_path):
    (tmp_path / "src").mkdir()
    with pytest.raises(testrun.TestEnvironmentUnavailable) as exc:
        testrun.discover_failing_tests(str(tmp_path))
    assert "no test runner was detected" in str(exc.value)


def test_cli_prints_the_environment_gap_with_its_own_code():
    from codna.cli import _structured_error

    payload = _structured_error(testrun.TestEnvironmentUnavailable("x.", details={"tail": "t"}))
    assert payload["error"]["code"] == "test_environment_unavailable"
    assert payload["error"]["details"] == {"tail": "t"}
