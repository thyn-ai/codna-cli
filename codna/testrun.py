"""Safe failing-test discovery for ``codna fix --tests``.

Runs the repo's test suite inside the hardened :class:`~codna.sandbox.Sandbox`: a
**credential-scrubbed env** (no write tokens AND no provider/cloud/engine keys — untrusted repo code
must see no secrets), a fixed cwd, a bounded timeout, and network denial **actually enforced** via the
kernel backend (bwrap/unshare) where available. When fail-closed egress is required
(``privacy.egress``) but no enforcer exists, it refuses rather than run open. Failing ids are parsed
structurally from a JUnit report so they keep the ``classname::name`` shape the engine expects
(identical to ``--from-junit``).

Which command runs (:func:`resolve_test_command`): ``--test-cmd`` > ``fix.test_command`` in the
repository's own ``codna.yaml`` > a runner the repository's tooling declares (``pixi run test``,
``uv run pytest``) > ``pytest``. When THIS environment cannot run the tests at all --- the runner is
not installed, pytest cannot import the test modules, no runner is detectable --- that is
:class:`TestEnvironmentUnavailable`, a property of the sandbox and never of the change under test:
the CLI prints it with its own error code and the GitHub App ends the ``codna fix`` check neutral.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import tempfile

# JUnit reports come from an untrusted repo's test run, so they are parsed with defusedxml:
# entity declarations (XXE / billion-laughs) are rejected instead of expanded.
import defusedxml.ElementTree as ElementTree

from .netjail import default_network_backend
from .sandbox import Sandbox, scrub_env

CONFIG_KEY = "fix.test_command"


class TestEnvironmentUnavailable(RuntimeError):
    """The repository's tests cannot run in THIS environment.

    A property of the sandbox (the hosted App image ships pytest and no other toolchain), not of the
    commit: on thyn-ai/mojo-kernels#1 (2026-09-19) bare pytest could not import wrappers that live in
    the repo's pixi environment, and the ``codna fix`` check went red over a pull request that had
    nothing wrong with it. Carries a stable ``code`` so ``codna`` prints a structured error the App
    turns into a neutral check (nothing to fix here, nothing to retry) with the way out in the text.
    """

    code = "test_environment_unavailable"

    def __init__(self, reason: str, *, command_known: bool = False, details: dict | None = None):
        # command_known: the command itself is settled (configured, or detected from the repo's own
        # tooling) and what is missing is the toolchain -- so do not ask for the command again.
        hint = (
            "Run `codna fix --tests` where that toolchain is installed, or hand Codna CI's own "
            "report with --from-junit."
            if command_known else
            f"Tell Codna how this repository's tests run with `{CONFIG_KEY}` in its codna.yaml "
            "(for example `pixi run test`), pass --test-cmd, or hand Codna CI's own report with "
            "--from-junit."
        )
        super().__init__(f"Codna could not run this repository's tests in its sandbox: {reason} {hint}")
        self.details = details or {}

# Untrusted repo test code (pytest collection imports conftest/test modules; --test-cmd is arbitrary
# shell) runs in this sandbox, so it must NOT see ANY credential — scrub_env only strips GitHub write
# tokens, which is not enough. Remove every var that looks like a secret (provider/cloud/engine keys).
_SECRET_NAME_RE = re.compile(
    r"(_API_KEY|_TOKEN|_SECRET|_ACCESS_KEY|_PASSWORD|_CREDENTIALS?|_PRIVATE_KEY|_SESSION)$"
    r"|^(AWS_|AZURE_|GCP_|GOOGLE_|ANTHROPIC_|OPENAI_|GEMINI_|GROQ_|MISTRAL_|OPENROUTER_|XAI_|COHERE_|"
    r"HF_|HUGGINGFACE_|CURSOR_|CODNA_|ALGENTA_|TELYS_|DE_|SUPABASE_|FLY_|CLOUDFLARE_|GITBOOK_)",
    re.IGNORECASE,
)


def _scrub_secrets(env: dict) -> dict:
    """Strip credential-looking vars by name (on top of scrub_env's write-token removal)."""
    return {k: v for k, v in env.items() if not _SECRET_NAME_RE.search(k)}


def parse_junit_failures(path: str) -> list[str]:
    """Failing/erroring test ids as ``classname::name`` from a JUnit XML report.

    Raises ``defusedxml.DefusedXmlException`` (a ``ValueError``) for a report that declares
    entities -- the callers surface that as a bad-input error rather than expanding it.
    """
    root = ElementTree.parse(os.path.expanduser(path)).getroot()
    failing: list[str] = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            name = case.get("name", "")
            cls = case.get("classname", "")
            failing.append(f"{cls}::{name}" if cls else name)
    return failing


# Depth-1 subdirectories never worth scanning for pytest config (never contain the repo's own
# tests, and some can be large enough to make even a shallow os.listdir a waste).
_SKIP_SUBDIRS = frozenset({
    "node_modules", "venv", ".venv", "dist", "build", "vendor", "__pycache__", "site-packages",
})


def _has_pytest_config(d: str) -> bool:
    if os.path.isfile(os.path.join(d, "pytest.ini")):
        return True
    if os.path.isdir(os.path.join(d, "tests")):
        return True
    pyproject = os.path.join(d, "pyproject.toml")
    if os.path.isfile(pyproject):
        try:
            import tomllib

            with open(pyproject, "rb") as fh:
                if "pytest" in (tomllib.load(fh).get("tool") or {}):
                    return True
        except Exception:  # noqa: BLE001
            pass
    cfg = os.path.join(d, "setup.cfg")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as fh:
                if "[tool:pytest]" in fh.read():
                    return True
        except OSError:
            pass
    return False


def _detect_pytest(repo_dir: str) -> str | None:
    """Return the directory pytest config lives in — ``repo_dir`` itself, or a depth-1
    subdirectory of it — or ``None`` if neither has any.

    Monorepo-aware: a repo's actual Python package (and its pytest.ini / tests/ / pyproject.toml
    [tool.pytest]) commonly lives one level down, in a subdirectory like ``cli/`` or ``python/``,
    rather than at the repo root. A root-only check would conclude "no test runner" despite a
    real, working pytest setup one level down; only searching depth-1 (not an arbitrary recursive
    walk) keeps this bounded and avoids false positives from an unrelated pytest config buried in
    a vendored/example subtree.

    Raises :class:`RuntimeError` when MULTIPLE depth-1 subdirectories each have their own pytest
    config and the root has none — codna's OWN repo is exactly this shape (``bench/``, ``cli/``,
    and ``python/`` are all independently valid pytest roots). Silently picking one (e.g. the
    alphabetically-first) would run an arbitrary subset of the repo's tests and could report
    "tests pass" while the actual failing suite never ran — a clear, actionable error is safer
    than a guess here.
    """
    if _has_pytest_config(repo_dir):
        return repo_dir
    try:
        entries = sorted(os.listdir(repo_dir))
    except OSError:
        return None
    candidates = [
        os.path.join(repo_dir, name)
        for name in entries
        if not name.startswith(".")
        and name not in _SKIP_SUBDIRS
        and os.path.isdir(os.path.join(repo_dir, name))
        and _has_pytest_config(os.path.join(repo_dir, name))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(os.path.basename(c) for c in candidates)
        raise TestEnvironmentUnavailable(
            f"pytest config lives in several subdirectories ({names}) and none at the repo root, so "
            "it is ambiguous which suite to run (pass --test-cmd \"<your test command>\" writing "
            "JUnit to $CODNA_JUNIT)."
        )
    return None


def repo_test_command(repo_dir: str) -> str | None:
    """``fix.test_command`` from the repository's OWN ``codna.yaml`` / ``.codna.yaml`` -- the checkout
    being fixed, not the operator's cwd (the same lookup ``review_findings.load_review_config`` does
    for the ``review:`` block). None when absent. Malformed YAML or a non-string value fails closed
    with :class:`~codna.config_file.ConfigError`, like every other codna.yaml mistake."""
    from .config_file import _SEARCH, ConfigError, _load

    for name in _SEARCH:
        path = os.path.join(repo_dir, name)
        if os.path.isfile(path):
            break
    else:
        return None
    fix = _load(path).get("fix")
    if fix is None:
        return None
    if not isinstance(fix, dict):
        raise ConfigError(f"{path}: `fix` must be a mapping")
    command = fix.get("test_command")
    if command is None:
        return None
    if not isinstance(command, str) or not command.strip():
        raise ConfigError(f"{path}: {CONFIG_KEY} must be a non-empty string")
    return command.strip()


def _pixi_tasks(repo_dir: str) -> tuple[dict, str] | None:
    """The pixi ``[tasks]`` table and the file it came from (``pixi.toml``, else a pyproject.toml
    with a ``[tool.pixi]`` section), or None when the repository is not pixi-managed."""
    import tomllib

    for name, keys in (("pixi.toml", ("tasks",)), ("pyproject.toml", ("tool", "pixi", "tasks"))):
        path = os.path.join(repo_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if name == "pyproject.toml" and not isinstance((data.get("tool") or {}).get("pixi"), dict):
            continue
        node: object = data
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        return (node if isinstance(node, dict) else {}), name
    return None


def detect_test_command(repo_dir: str) -> tuple[str, str] | None:
    """``(command, source_file)`` when the repository's own tooling declares how its tests run.

    Only signals that name BOTH the runner and the test entrypoint count -- a guess that runs the
    wrong thing is worse than the pytest default:

    * pixi: a ``test`` task in ``pixi.toml`` (or ``[tool.pixi.tasks]``) -> ``pixi run test``. The
      task carries the env the suite needs (toolchain, PYTHONPATH, oracles) that bare pytest lacks.
    * uv: a ``uv.lock`` that locks pytest, next to pytest config -> ``uv run pytest``.

    JavaScript runners (``pnpm test`` and friends) are deliberately not detected: they never produce
    the JUnit report that separates "tests failed" from "node_modules missing", and the install step
    they need is exactly what the network-denied sandbox forbids.
    """
    pixi = _pixi_tasks(repo_dir)
    if pixi is not None and "test" in pixi[0]:
        return "pixi run test", pixi[1]
    uv_lock = os.path.join(repo_dir, "uv.lock")
    if os.path.isfile(uv_lock) and _has_pytest_config(repo_dir):
        try:
            with open(uv_lock, encoding="utf-8", errors="replace") as fh:
                locked = fh.read()
        except OSError:
            locked = ""
        if re.search(r'^name = "pytest"$', locked, re.M):
            return "uv run pytest", "uv.lock"
    return None


def resolve_test_command(repo_dir: str, test_cmd: str | None = None) -> tuple[str, str, bool] | None:
    """``(command, source, explicit)`` for the shell command that runs this repository's tests, or
    None for the pytest default.

    Precedence: ``--test-cmd`` > the repository's ``fix.test_command`` > :func:`detect_test_command`.
    ``explicit`` is True when a person wrote the command (flag or config): its exit status is then the
    truth about the tests. A detected command is a well-founded guess, trusted only once pytest has
    actually produced a report.
    """
    if test_cmd:
        return test_cmd, "--test-cmd", True
    configured = repo_test_command(repo_dir)
    if configured:
        return configured, f"codna.yaml {CONFIG_KEY}", True
    detected = detect_test_command(repo_dir)
    if detected:
        return detected[0], detected[1], False
    return None


def junit_has_collection_errors(path: str) -> bool:
    """True when pytest could not even import a test module (``<error message="collection failure">``)."""
    try:
        root = ElementTree.parse(path).getroot()
    except Exception:  # noqa: BLE001
        return False
    return any(str(err.get("message", "")).lower().startswith("collection failure") for err in root.iter("error"))


def discover_failing_tests(repo_dir: str, test_cmd: str | None = None, *, timeout_seconds: int = 1800):
    """``(issue_text | None, [failing_ids])`` — same contract as :func:`fix_inputs.from_junit`.

    Runs the command :func:`resolve_test_command` picks (pytest by default) inside a network-denied
    sandbox with a scrubbed env, writing a JUnit report we parse for per-test ids. Raises
    :class:`TestEnvironmentUnavailable` when this environment cannot run the tests, and a plain
    :class:`RuntimeError` for operator-side problems (bad directory, timeout, egress policy). A shell
    command may write its JUnit report to ``$CODNA_JUNIT``; any pytest it reaches does so on its own
    (``PYTEST_ADDOPTS``), otherwise only pass/fail is known.
    """
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    if not os.path.isdir(repo_dir):
        raise RuntimeError(f"--tests needs a local repo directory; {repo_dir} is not a directory")

    junit = os.path.join(tempfile.mkdtemp(prefix="codna-junit-"), "report.xml")
    # Scrub write tokens (scrub_env) AND every credential-looking var (_scrub_secrets) so untrusted
    # repo test code can't read the operator's provider/cloud/engine keys. Set CODNA_JUNIT AFTER
    # scrubbing (its CODNA_ prefix would otherwise be stripped).
    env = _scrub_secrets(scrub_env(dict(os.environ)))
    env["CODNA_JUNIT"] = junit  # custom test commands can write their report here
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # prevent test runs from dirtying repos with __pycache__

    run_dir = repo_dir  # overridden below only when pytest config is found in a subdirectory
    resolved = resolve_test_command(repo_dir, test_cmd)
    command, source, explicit = resolved if resolved else (None, None, False)
    if command:
        runner = command.split()[0]
        if not explicit and shutil.which(runner) is None:
            # A detected runner we know by name: say so before running anything, instead of letting
            # the shell's exit 127 stand in for "the tests failed".
            raise TestEnvironmentUnavailable(
                f"`{command}` (detected from {source}) needs `{runner}`, which is not installed here.",
                command_known=True,
            )
        argv = ["/bin/sh", "-c", command]
        # pixi tasks, `uv run pytest`, make targets: none pass --junitxml themselves, so ask any
        # pytest the command reaches to write the report Codna parses. Additive to the repo's own
        # PYTEST_ADDOPTS, inert for a non-pytest runner.
        env["PYTEST_ADDOPTS"] = f"{env.get('PYTEST_ADDOPTS', '')} --junitxml={junit}".strip()
    else:
        pytest_dir = _detect_pytest(repo_dir)
        if pytest_dir:
            if importlib.util.find_spec("pytest") is None:
                raise TestEnvironmentUnavailable(
                    "pytest config was found but pytest is not installed here.", command_known=True
                )
            argv = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"]
            run_dir = pytest_dir  # e.g. cli/ when the repo root itself has no pytest config
        else:
            raise TestEnvironmentUnavailable(
                "no test runner was detected (no pytest.ini / tests/ / pyproject [tool.pytest] at the "
                "repo root or one level down, no pixi `test` task, no uv.lock with pytest)."
            )

    # Actually ENFORCE network denial (bwrap/unshare on Linux CI) — not just record it — so untrusted
    # test code can't exfiltrate. When fail-closed egress is required (privacy.egress) but no kernel
    # enforcer is available (e.g. macOS dev host), refuse rather than run with unenforced network.
    backend = default_network_backend()
    if os.environ.get("CODNA_REQUIRE_EGRESS_DENY") == "1" and backend is None:
        raise RuntimeError(
            "privacy.egress=fail-closed requires kernel-level egress denial (bwrap/unshare), which is "
            "unavailable here — refusing to run `--tests` with unenforced network."
        )
    # pytest exits 1 when tests fail — that is the expected/accepted case here (not an error).
    result = Sandbox(network="deny", timeout_seconds=timeout_seconds, env=env, network_backend=backend).run(
        argv, cwd=run_dir, accepted_exit_codes=(0, 1)
    )
    if result.timed_out:
        raise RuntimeError(f"test run timed out after {timeout_seconds}s")

    tail = (result.stderr or result.stdout or "")[-1500:]
    if command and result.returncode == 127:
        # POSIX sh: "command not found". `pixi run test` on an image without pixi is this, and it
        # is not a failing test whatever the command's origin.
        raise TestEnvironmentUnavailable(
            f"`{command}` (from {source}) exited 127 (command not found), so its runner is not "
            "installed here.",
            command_known=True, details={"tail": tail},
        )
    if os.path.isfile(junit) and junit_has_collection_errors(junit):
        # An import error is not a failing test: "fixing" it would mean editing the repo until its
        # tests import in THIS environment (the webhook image has pytest and nothing else).
        raise TestEnvironmentUnavailable(
            "pytest could not import the test modules (collection errors, usually missing "
            "dependencies), and an import error is not a failing test.",
            command_known=command is not None,
        )
    if command and not explicit and result.returncode != 0 and not os.path.isfile(junit):
        # A detected command that died before any pytest ran (a build step, an environment that
        # needs the network the sandbox denies): no evidence about the tests, so no "failing tests"
        # for the agent to chase -- that is exactly how a refused test-only patch got produced.
        raise TestEnvironmentUnavailable(
            f"`{command}` (detected from {source}) exited {result.returncode} before pytest produced "
            "a report, so that toolchain is not runnable here.",
            command_known=True, details={"tail": tail},
        )
    failing = parse_junit_failures(junit) if os.path.isfile(junit) else []
    if failing:
        return f"{len(failing)} failing test(s): " + "; ".join(failing[:8]), failing
    if result.returncode not in (0,):
        tail = (result.stdout or result.stderr or "")[-1500:]
        return ("tests are failing (no per-test JUnit ids were produced):\n" + tail), []
    return None, []
