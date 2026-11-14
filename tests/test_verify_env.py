"""The PATH verification runs with — the defect where a correct repository verified as `127`.

── THE MEASURED FAILURE ─────────────────────────────────────────────────────────────────────────

Run ``run-20261113T225225Z`` built a working three-file app and verified it three times. Every attempt:

    exit 127  /bin/sh: node: command not found

while the same command in the same directory, run by the user, passed 3/3. The worker had been started
by a launcher with a minimal PATH; ``node`` lives in a directory only a LOGIN shell adds. ``verify_task``
handed ``scrub_env(os.environ)`` to the sandbox, and ``scrub_env`` is a credential FILTER — it correctly
leaves PATH alone — so the sandbox searched the worker's minimal PATH and reported a correct repository
as failing.

── HOW THESE TESTS STAY PORTABLE ────────────────────────────────────────────────────────────────

The fix under test exists BECAUSE assuming one machine's layout is wrong: `/opt/homebrew/bin` is not
where a Linux user's tools live, and `/bin/zsh` need not exist there at all. A test that hardcoded
either would reproduce the defect inside the tests, and one did: asserting
``resolved.shell == "/bin/zsh"`` passed on macOS and failed on `ubuntu-latest`, where ``$SHELL`` is
``/bin/bash`` and ``/bin/zsh`` is absent — so the resolver, correctly, never considered zsh and
reported the shell it actually asked.

Every test below therefore OWNS both halves of the probe: the candidate list (``_probe`` pins
``_shell_candidates``) and the shell's answer (``_fake_shell``). Nothing reads the host's shell, its
directory layout, or its dotfiles, so the same assertions run identically on macOS and Linux. The two
exceptions are deliberate and still portable: ``test_the_env_shell_is_used_only_when_it_is_a_real_
executable`` builds its own shells under ``tmp_path``, and ``test_the_probe_works_on_a_real_posix_shell``
uses ``/bin/sh`` — the one shell POSIX guarantees — asserting the probe FORMAT survives a real shell's
startup files rather than a particular shell path.

── WHAT THIS FILE PINS ──────────────────────────────────────────────────────────────────────────

* The login shell's entries REACH the verify PATH (the defect itself).
* The worker's own entries keep their precedence, so existing resolutions do not change underneath a user.
* A probe that fails or times out still yields a usable PATH, and says why — never empty, never a raise.
* A login shell that cannot be asked does not silently become "your tests failed".
* ``CODNA_VERIFY_PATH`` replaces the probe (and an empty one disables it) for hermetic CI.
* ``scrub_env`` still strips credentials: the security boundary this change sits beside, not across.

Each test is written to be RED against the pre-fix code and GREEN against the fix:
`resolve_verify_environment` did not exist before, so the pre-fix behaviour is reconstructed where it
matters (a PATH with no login entries reaches the sandbox) rather than asserted against a mock of itself.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from codna import verify_env
from codna.sandbox import PrivilegeSeparationError, Sandbox, scrub_env
from codna.verify_env import (
    EXPLICIT_ENV,
    PROBE_TIMEOUT_S,
    login_shell_path,
    merge_paths,
    resolve_verify_environment,
    reset_cache,
)

#: The worker's own PATH in the measured failure: everything a launcher gives a daemon.
MINIMAL_PATH = "/usr/bin:/bin"

#: The shell this file injects. A NAME only: whether it exists on the host is exactly what these tests
#: must not depend on (`ubuntu-latest` has no `/bin/zsh`). Every test that names it also pins the
#: candidate list with `_probe`, so the string is fixture data and is never stat'ed or executed.
FAKE_SHELL = "/bin/zsh"

#: Illustrative toolchain directories. Deliberately NOT Homebrew: no test may assert that a location one
#: platform happens to use is universal. These are strings the fake login shell reports; nothing reads
#: them from disk.
TOOLCHAIN_BIN = "/opt/example-toolchain/bin"
USER_LOCAL_BIN = "/home/example/.local/bin"


@pytest.fixture(autouse=True)
def _clean_cache():
    """The resolved environment is cached for the process lifetime, so every test starts from zero."""
    reset_cache()
    yield
    reset_cache()


def _probe(monkeypatch, *shells: str) -> None:
    """Pin the candidate list the resolver probes, in order.

    `_shell_candidates` derives its answer from the HOST: `$SHELL` counts only when it is an absolute,
    executable file, then the platform's preferred shells that exist. That is correct production
    behaviour and exactly what makes a test non-deterministic when it is left implicit — on
    `ubuntu-latest` a test that sets ``SHELL=/bin/zsh`` gets ``/bin/bash`` back instead, because zsh is
    not installed there. Injecting the list makes the shell under test data, not the runner's layout.
    """
    monkeypatch.setattr(verify_env, "_shell_candidates", lambda env: tuple(shells))


def _shell_result(
    shell: str, path: str, *, returncode: int = 0, prefix: str = "", suffix: str = ""
) -> subprocess.CompletedProcess:
    """A `CompletedProcess` shaped like a login shell answering the PATH probe."""
    script_out = f"{verify_env._MARK_BEGIN}{path}{verify_env._MARK_END}"
    return subprocess.CompletedProcess(
        args=[shell, "-lc", verify_env._PROBE_SCRIPT],
        returncode=returncode,
        stdout=prefix + script_out + suffix,
        stderr="",
    )


def _fake_shell(
    *, path: str, seen: list | None = None, **result_kwargs
) -> "verify_env.Runner":
    """A probe runner that reports `path`, recording the argv it was asked to run.

    The reported SHELL is `argv[0]` — the shell that was actually invoked — because that is what a real
    probe does (`login_shell_path` returns the shell it got an answer from). Deriving it from the argv
    rather than a constant keeps `resolved.shell` a property of the injected candidate list.
    """

    def run(argv, **kwargs):
        if seen is not None:
            seen.append((argv, kwargs))
        return _shell_result(argv[0], path, **result_kwargs)

    return run


def _failing_shell(exc: Exception):
    def run(argv, **kwargs):
        raise exc

    return run


# -- the defect ---------------------------------------------------------------------------------


def test_the_verify_path_includes_the_login_shell_entries(monkeypatch):
    """THE DEFECT. A worker PATH without the toolchain, and a login shell that has it.

    This is the measured shape: `/usr/bin:/bin` cannot find `node`, and the login shell names the
    directory where it lives. Before the fix the sandbox received the worker PATH unchanged, which is
    why the correct repository exited 127.
    """
    _probe(monkeypatch, FAKE_SHELL)
    seen: list = []
    resolved = resolve_verify_environment(
        base_env={"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL},
        run=_fake_shell(
            path=f"{USER_LOCAL_BIN}:{TOOLCHAIN_BIN}:{MINIMAL_PATH}:/usr/sbin", seen=seen
        ),
    )

    entries = resolved.path.split(os.pathsep)
    # The exact merge, not just membership: worker entries first and in order, then the login entries
    # that are new, deduplicated.
    assert entries == ["/usr/bin", "/bin", USER_LOCAL_BIN, TOOLCHAIN_BIN, "/usr/sbin"], entries
    assert resolved.source == "login-shell"
    # The shell that ANSWERED the probe, which the test injected — not the host's shell.
    assert resolved.shell == FAKE_SHELL
    assert [argv[0] for argv, _ in seen] == [FAKE_SHELL], seen
    # The verifiable consequence, not just a string: the prepared env can find a binary that only the
    # login shell knows about. This is what "node: command not found" becomes.
    assert TOOLCHAIN_BIN in resolved.env["PATH"].split(os.pathsep)
    assert resolved.env["PATH"] == resolved.path
    assert "login-shell PATH" in resolved.note


def test_the_worker_path_keeps_precedence_over_the_login_shell(monkeypatch):
    """Additive, never overriding: an explicitly injected PATH must still resolve first.

    A venv, a CI image, or the packaged app's own runtime puts its directories on PATH on purpose. Had
    the login entries been prepended, every existing user's `python3`/`node` would silently change —
    a fix that breaks the case that was working.
    """
    _probe(monkeypatch, FAKE_SHELL)
    resolved = resolve_verify_environment(
        base_env={"PATH": f"/venv/bin:{MINIMAL_PATH}", "SHELL": FAKE_SHELL},
        run=_fake_shell(path=f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}"),
    )
    entries = resolved.path.split(os.pathsep)
    assert entries[:2] == ["/venv/bin", "/usr/bin"], entries
    assert entries.index(TOOLCHAIN_BIN) > entries.index("/venv/bin")


def test_duplicate_entries_are_not_repeated(monkeypatch):
    """A login shell already containing the worker's directories must not grow PATH."""
    _probe(monkeypatch, FAKE_SHELL)
    resolved = resolve_verify_environment(
        base_env={"PATH": f"{MINIMAL_PATH}:{TOOLCHAIN_BIN}", "SHELL": FAKE_SHELL},
        run=_fake_shell(path=f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}:/usr/sbin"),
    )
    entries = resolved.path.split(os.pathsep)
    assert entries == ["/usr/bin", "/bin", TOOLCHAIN_BIN, "/usr/sbin"], entries
    assert merge_paths(MINIMAL_PATH, f"{MINIMAL_PATH}:{TOOLCHAIN_BIN}") == f"{MINIMAL_PATH}:{TOOLCHAIN_BIN}"


def test_a_bash_only_candidate_list_resolves_identically(monkeypatch):
    """The Linux shape of the same run: no zsh on the machine, so bash answers and is reported.

    `ubuntu-latest` has no `/bin/zsh`, and this is the case that failed CI: the resolver reported
    ``/bin/bash`` (correctly — it is what answered) while the test demanded zsh. Injecting bash-only
    candidates asserts the behaviour the fix promises — whichever shell answers is the one recorded —
    without needing a platform to be missing anything.
    """
    _probe(monkeypatch, "/bin/bash")
    resolved = resolve_verify_environment(
        base_env={"PATH": MINIMAL_PATH, "SHELL": "/bin/bash"},
        run=_fake_shell(path=f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}"),
    )
    assert resolved.shell == "/bin/bash"
    assert resolved.source == "login-shell"
    assert resolved.path == f"{MINIMAL_PATH}:{TOOLCHAIN_BIN}"
    assert "/bin/bash" in resolved.note
    assert "bash" in resolved.describe()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")
def test_the_probe_works_on_a_real_posix_shell(monkeypatch):
    """The probe format, exercised against a REAL shell — not only a fake `CompletedProcess`.

    `/bin/sh` is the one shell POSIX guarantees to exist, so this reads no platform's layout beyond the
    standard. It is the check that `-lc` plus the marker script actually survives a shell's own startup
    files: `/etc/profile` is free to rewrite `PATH`, which is the reason this asserts a usable PATH
    rather than a byte-identical one.
    """
    _probe(monkeypatch, "/bin/sh")
    found = login_shell_path({"PATH": MINIMAL_PATH, "SHELL": "/bin/sh"})
    assert found.reason is None, found
    assert found.shell == "/bin/sh"
    assert found.path, "a real login shell must report a non-empty PATH"
    entries = found.path.split(os.pathsep)
    assert "/bin" in entries or "/usr/bin" in entries, entries


def test_the_env_shell_is_used_only_when_it_is_a_real_executable(tmp_path):
    """The RULE behind the CI failure, without naming a platform's shell.

    `$SHELL` is preferred because it is the shell whose dotfiles define this user's PATH — but only if
    it can actually be run. A launcher can hand a worker a `$SHELL` pointing at an interpreter that is
    not installed (a container image, a minimal CI job, a stale dotfile), and probing it would waste the
    timeout budget before falling through.

    Both shells here are created by the test, so the assertion is about the RULE and not about which
    shells the runner ships. A previous version of this test read the host's `/bin/zsh`, which is the
    same assumption that failed CI.
    """
    executable = tmp_path / "login-shell"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    not_executable = tmp_path / "not-executable"
    not_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    missing = tmp_path / "no-such-shell"

    # An executable $SHELL is asked first: it is the shell whose dotfiles this user actually inherits.
    assert verify_env._shell_candidates({"SHELL": str(executable)})[0] == str(executable)
    # A $SHELL that is absent, or present but not executable, is skipped rather than probed and failed.
    assert str(missing) not in verify_env._shell_candidates({"SHELL": str(missing)})
    assert str(not_executable) not in verify_env._shell_candidates({"SHELL": str(not_executable)})
    # A relative $SHELL is not a login shell a launcher can be trusted to mean; skip it too.
    assert "zsh" not in verify_env._shell_candidates({"SHELL": "zsh"})
    # Whatever is probed must be executable, or the probe is a guaranteed failure before it starts.
    for candidate in verify_env._shell_candidates({"SHELL": str(missing)}):
        assert os.access(candidate, os.X_OK), candidate


# -- failure modes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "runner", "expected_reason"),
    [
        ("times-out", _failing_shell(subprocess.TimeoutExpired(cmd=FAKE_SHELL, timeout=PROBE_TIMEOUT_S)), "did not answer"),
        (
            "exits-non-zero",
            lambda argv, **kw: _shell_result(argv[0], "", returncode=1),
            "exited 1",
        ),
        ("no-marker", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "a banner, no markers", ""), "no PATH marker"),
        ("empty-path", lambda argv, **kw: _shell_result(argv[0], ""), "empty PATH"),
        ("cannot-start", _failing_shell(OSError("no such file")), "could not be started"),
    ],
)
def test_a_failed_probe_keeps_a_usable_path_and_records_why(monkeypatch, label, runner, expected_reason):
    """FAIL SAFELY: a probe that cannot answer is a fallback with a reason, never an empty PATH.

    The failure mode this rules out is the one that turns an environment problem into a green-but-wrong
    verdict: an empty PATH makes every command `127`, which reads as "the repository's tests fail".
    """
    _probe(monkeypatch, FAKE_SHELL)
    resolved = resolve_verify_environment(
        base_env={"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL}, run=runner
    )
    assert resolved.env["PATH"] == MINIMAL_PATH, label
    assert resolved.source == "process"
    assert resolved.shell is None
    assert expected_reason in resolved.note, (label, resolved.note)


def test_a_worker_with_no_path_at_all_still_gets_one(monkeypatch):
    """The last resort. `env -i` in a launcher leaves no PATH; `os.defpath` is the honest answer."""
    _probe(monkeypatch, FAKE_SHELL)
    resolved = resolve_verify_environment(
        base_env={"SHELL": FAKE_SHELL},
        run=lambda argv, **kw: _shell_result(argv[0], ""),
    )
    assert resolved.env["PATH"] == os.defpath
    assert "no PATH" in resolved.note


def test_the_probe_is_bounded_and_runs_on_a_scrubbed_environment(monkeypatch):
    """The probe itself must not be a stall, and must not carry a credential into the user's dotfiles."""
    _probe(monkeypatch, FAKE_SHELL)
    seen: list = []
    resolve_verify_environment(
        base_env={"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL, "GITHUB_TOKEN": "ghp_" + "a" * 36},
        run=_fake_shell(path=TOOLCHAIN_BIN, seen=seen),
    )
    argv, kwargs = seen[0]
    assert argv[0] == FAKE_SHELL and argv[1] == "-lc"
    # A login-shell probe asks for $PATH; asserting the script is what makes `-lc` load-bearing rather
    # than decorative.
    assert argv[2] == verify_env._PROBE_SCRIPT
    assert kwargs["timeout"] == PROBE_TIMEOUT_S
    assert "GITHUB_TOKEN" not in kwargs["env"]


def test_a_banner_on_stdout_does_not_corrupt_the_path(monkeypatch):
    """Dotfiles print things. The markers make the answer extractable anyway."""
    _probe(monkeypatch, FAKE_SHELL)
    found = login_shell_path(
        {"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL},
        run=lambda argv, **kw: _shell_result(
            argv[0], f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}", prefix="welcome!\n", suffix="\nbye\n"
        ),
    )
    assert found.path == f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}"
    assert found.shell == FAKE_SHELL
    assert found.reason is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")
def test_the_probe_script_is_inert_whatever_the_path_contains(tmp_path):
    """The probe is a shell WORD, so the thing it must survive is the shell's own expansion rules.

    `$PATH` reaches `printf` as an ARGUMENT (``printf '<mark>'%s'<mark>' "$PATH"``), never spliced into
    the script text: the script is a constant, so nothing in a PATH can be read as shell syntax. This
    runs the real script on a real `/bin/sh` for the shapes a directory name really can contain — an
    apostrophe, a percent sign, and a command substitution — and asserts the SIDE EFFECT is absent, not
    only the string: a PATH entry is data, so `$(...)` inside it must never execute.

    ``-c`` rather than ``-lc`` on purpose: `/etc/profile` is free to rewrite PATH (macOS `path_helper`,
    a Linux image's own default), which would make this an assertion about the runner's profile rather
    than about the probe's quoting. Login behaviour is covered by
    `test_the_probe_works_on_a_real_posix_shell`.
    """
    witness = tmp_path / "witness"
    hostile = [
        f"{tmp_path}/it's a dir/bin",
        f"{tmp_path}/100%/bin",
        f"{tmp_path}/$(touch {witness})/bin",
        f"{tmp_path}/`touch {witness}`/bin",
        f"{tmp_path}/a b/bin",
    ]
    env = {"PATH": os.pathsep.join([*hostile, MINIMAL_PATH]), "HOME": str(tmp_path)}
    assert "$" not in verify_env._PROBE_SCRIPT.split('"')[0], "the script text must not interpolate PATH"

    proc = subprocess.run(
        ["/bin/sh", "-c", verify_env._PROBE_SCRIPT], capture_output=True, text=True, env=env
    )

    assert proc.returncode == 0, proc
    start = proc.stdout.find(verify_env._MARK_BEGIN)
    end = proc.stdout.find(verify_env._MARK_END, start + len(verify_env._MARK_BEGIN))
    assert start >= 0 and end >= 0, proc.stdout
    reported = proc.stdout[start + len(verify_env._MARK_BEGIN) : end]
    assert reported == env["PATH"], reported
    assert not witness.exists(), "a PATH entry must never be executed as shell syntax"


def test_no_login_shell_at_all_is_a_reason_not_a_crash(monkeypatch):
    """Platforms with none of the known shells: still a value, still explained."""
    _probe(monkeypatch)  # empty candidate list
    resolved = resolve_verify_environment(base_env={"PATH": MINIMAL_PATH}, run=_fake_shell(path="/nope"))
    assert resolved.env["PATH"] == MINIMAL_PATH
    assert "no login shell was found" in resolved.note


# -- explicit configuration ---------------------------------------------------------------------


def test_the_explicit_override_replaces_the_probe(monkeypatch):
    """`CODNA_VERIFY_PATH` is the escape hatch: it is used INSTEAD of asking a shell, not as a prefix.

    Needed where no login shell exists but a toolchain does — a container, a CI job — and where a
    hermetic run must not read the user's dotfiles at all.
    """
    _probe(monkeypatch, FAKE_SHELL)
    seen: list = []
    resolved = resolve_verify_environment(
        base_env={"PATH": MINIMAL_PATH, EXPLICIT_ENV: "/nix/store/bin", "SHELL": FAKE_SHELL},
        run=_fake_shell(path=TOOLCHAIN_BIN, seen=seen),
    )
    assert resolved.path == f"{MINIMAL_PATH}:/nix/store/bin"
    assert resolved.source == "explicit"
    assert seen == [], "the login-shell probe must not run when the override is set"
    assert EXPLICIT_ENV in resolved.note


def test_an_empty_override_disables_the_probe(monkeypatch):
    """Empty-but-set means "do not look": verify runs on the worker's PATH, as a hermetic CI wants."""
    _probe(monkeypatch, FAKE_SHELL)
    seen: list = []
    resolved = resolve_verify_environment(
        base_env={"PATH": MINIMAL_PATH, EXPLICIT_ENV: "", "SHELL": FAKE_SHELL},
        run=_fake_shell(path=TOOLCHAIN_BIN, seen=seen),
    )
    assert resolved.path == MINIMAL_PATH
    assert resolved.source == "explicit"
    assert seen == []


# -- cache --------------------------------------------------------------------------------------


def test_the_probe_runs_once_per_process(monkeypatch):
    """One shell per worker, not one per verify attempt — and the same environment either way.

    The cache is keyed on (override, PATH, shell candidates), so an unchanged worker cannot get two
    different PATHs minutes apart, which is the determinism the durable retry path depends on.
    """
    _probe(monkeypatch, FAKE_SHELL)
    calls: list = []

    def run(argv, **kwargs):
        calls.append(argv)
        return _shell_result(argv[0], f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}")

    base = {"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL}
    first = resolve_verify_environment(base_env=base, run=run)
    second = resolve_verify_environment(base_env=base, run=run)
    assert len(calls) == 1, calls
    assert first is second


def test_a_changed_path_reprobes(monkeypatch):
    """The key is the invalidation story: a different worker PATH must not reuse the old answer."""
    _probe(monkeypatch, FAKE_SHELL)
    calls: list = []

    def run(argv, **kwargs):
        calls.append(argv)
        return _shell_result(argv[0], f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}")

    resolve_verify_environment(base_env={"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL}, run=run)
    resolve_verify_environment(base_env={"PATH": "/srv/bin:/usr/bin", "SHELL": FAKE_SHELL}, run=run)
    assert len(calls) == 2, calls


def test_the_cached_environment_is_not_shared_by_reference(monkeypatch):
    """The sandbox gets a COPY, so a sandbox cannot poison the cache for every later attempt."""
    _probe(monkeypatch, FAKE_SHELL)
    base = {"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL}
    resolved = resolve_verify_environment(base_env=base, run=_fake_shell(path=TOOLCHAIN_BIN))
    handed = dict(resolved.env)
    handed["PATH"] = "/tampered"
    again = resolve_verify_environment(base_env=base, run=_fake_shell(path=TOOLCHAIN_BIN))
    assert again.env["PATH"] == resolved.env["PATH"]
    assert TOOLCHAIN_BIN in again.env["PATH"].split(os.pathsep)


def test_a_caller_cannot_mutate_the_cached_environment(monkeypatch):
    """The value handed out IS the cached value, so writing to it must be refused, not silently obeyed.

    The sandbox copies before it runs, so today's caller is safe. That is a convention held by one call
    site: a second caller that wrote `environment.env["PATH"] = ...` would corrupt every later verify in
    the worker, and nothing would say so until an unrelated attempt resolved a PATH nobody asked for.
    The mapping is read-only so the mistake raises where it is made.
    """
    _probe(monkeypatch, FAKE_SHELL)
    base = {"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL}
    resolved = resolve_verify_environment(base_env=base, run=_fake_shell(path=TOOLCHAIN_BIN))

    with pytest.raises(TypeError):
        resolved.env["PATH"] = "/tampered"  # type: ignore[index]

    again = resolve_verify_environment(base_env=base, run=_fake_shell(path=TOOLCHAIN_BIN))
    assert again.env["PATH"] == resolved.path
    assert TOOLCHAIN_BIN in again.env["PATH"].split(os.pathsep)
    # The copy the sandbox needs is still one call away, and it is a real mutable dict.
    assert isinstance(dict(resolved.env), dict)


def test_a_different_probe_runner_is_not_served_the_previous_answer(monkeypatch):
    """`run` is part of the cache key: the argument is honoured, not ignored once the cache is warm.

    A test that swaps the runner without calling `reset_cache()`, or an embedded runner that re-probes
    with its own executor, asks a DIFFERENT question about the same PATH. Returning the first answer
    would silently bypass the callable, and the caller would have no way to tell.
    """
    _probe(monkeypatch, FAKE_SHELL)
    calls: list[str] = []

    def first(argv, **kwargs):
        calls.append("first")
        return _shell_result(argv[0], f"{TOOLCHAIN_BIN}:{MINIMAL_PATH}")

    def second(argv, **kwargs):
        calls.append("second")
        return _shell_result(argv[0], f"{USER_LOCAL_BIN}:{MINIMAL_PATH}")

    base = {"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL}
    one = resolve_verify_environment(base_env=base, run=first)
    two = resolve_verify_environment(base_env=base, run=second)

    assert calls == ["first", "second"], calls
    assert one.path == f"{MINIMAL_PATH}:{TOOLCHAIN_BIN}"
    assert two.path == f"{MINIMAL_PATH}:{USER_LOCAL_BIN}"
    assert two is not one
    # The same runner still hits the cache: the key discriminates runners, it does not disable caching.
    again = resolve_verify_environment(base_env=base, run=first)
    assert again is one
    assert calls == ["first", "second"], calls


# -- the security boundary this sits BESIDE -----------------------------------------------------


def test_scrub_env_still_strips_credentials_from_the_verify_environment(monkeypatch):
    """GUARD. The fix adds PATH resolution around `scrub_env`; it must not weaken the filter.

    If a later change moved credential removal into the verify step "while we were in there", the
    untrusted repository's tests would see the worker's GitHub token. This test is the tripwire.
    """
    _probe(monkeypatch, FAKE_SHELL)
    ambient = {
        "PATH": MINIMAL_PATH,
        "GITHUB_TOKEN": "x",
        "CODNA_GITHUB_TOKEN": "y",
        "LEAKY": "github_pat_" + "b" * 30,
        "SAFE": "value",
    }
    scrubbed = scrub_env(dict(ambient))
    resolved = resolve_verify_environment(base_env=scrubbed, run=_fake_shell(path=TOOLCHAIN_BIN))

    assert "GITHUB_TOKEN" not in resolved.env
    assert "CODNA_GITHUB_TOKEN" not in resolved.env
    assert "LEAKY" not in resolved.env
    assert resolved.env["SAFE"] == "value"
    # And the sandbox would refuse the unscrubbed one, so the scrub is load-bearing rather than decorative.
    with pytest.raises(PrivilegeSeparationError):
        Sandbox(env=dict(ambient))


# -- end to end, still without a subprocess -----------------------------------------------------


def test_the_sandbox_reports_which_path_it_used(tmp_path):
    """A 127 must be diagnosable from the attempt alone. The note travels on the result."""
    sb = Sandbox(env={"PATH": os.environ.get("PATH", MINIMAL_PATH)}, env_note=f"login-shell via {FAKE_SHELL}: test")
    result = sb.run(["/bin/sh", "-c", "true"], cwd=str(tmp_path))
    assert result.returncode == 0
    assert result.env_note == f"login-shell via {FAKE_SHELL}: test"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")
def test_a_resolved_environment_finds_a_binary_that_the_worker_path_cannot(monkeypatch, tmp_path):
    """The mechanism, exercised with real subprocesses and no fixture repository.

    Two PATHs, one binary that exists only in the second directory. The worker PATH reports `127`; the
    resolved PATH runs it. This is exactly the measured failure and its fix, without needing `node`.
    """
    _probe(monkeypatch, FAKE_SHELL)
    tool_dir = tmp_path / "toolchain" / "bin"
    tool_dir.mkdir(parents=True)
    tool = tool_dir / "codna_fixture_tool"
    tool.write_text("#!/bin/sh\necho ran\n", encoding="utf-8")
    tool.chmod(0o755)

    worker_env = {"PATH": MINIMAL_PATH}
    before = Sandbox(env=worker_env).run(["/bin/sh", "-c", "codna_fixture_tool"], cwd=str(tmp_path))
    assert before.returncode == 127, before

    resolved = resolve_verify_environment(
        base_env={"PATH": MINIMAL_PATH, "SHELL": FAKE_SHELL},
        run=_fake_shell(path=f"{tool_dir}:{MINIMAL_PATH}"),
    )
    after = Sandbox(env=dict(resolved.env)).run(
        ["/bin/sh", "-c", "codna_fixture_tool"], cwd=str(tmp_path)
    )
    assert after.returncode == 0, after
    assert after.stdout.strip() == "ran"
