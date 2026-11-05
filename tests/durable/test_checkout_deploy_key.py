"""D6 deploy-key checkout: the key never reaches argv, a file, or a log — and cannot be skipped.

The properties, each one a failure mode:

* **The key is not in the command line.** `ps` on a shared host shows every argv. The checkout runs git
  with `GIT_SSH_COMMAND` pointing at a WRAPPER SCRIPT, and this test asserts the wrapper's path — not
  the key path — is what git sees, and that the key's BYTES appear in no argv.
* **The key file is private and transient.** `0600` inside a `mkdtemp` (`0700`), and unlinked when the
  clone finishes, including on failure. A crash must not leave a usable credential behind.
* **Host verification cannot be turned off.** `StrictHostKeyChecking=yes` with an explicit known-hosts
  file, defaulting to GitHub's published keys; an empty known-hosts value is a REFUSAL, not a
  downgrade.
* **A dirty destination is refused, never cleaned.** Deleting files in a path a caller handed us is how
  a mistake destroys a checkout.
* **An https remote is expressed as SSH, and anything unrecognised is refused.** Guessing a rewrite
  could push a branch to a host nobody intended.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from codna.durable.checkout import (
    DEPLOY_KEY_ENV,
    GITHUB_KNOWN_HOSTS,
    CheckoutError,
    DeployKey,
    checkout_repository,
    deploy_key_from_env,
    scm_url_for,
)

_FAKE_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gt\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)


# -- reading the key -----------------------------------------------------------------------


def test_no_key_configured_is_not_a_failure():
    """`None` means "the token path applies": a development host has no deploy key at all."""
    assert deploy_key_from_env({}, lookup=lambda _n: None) is None


def test_a_key_comes_from_the_environment_first():
    key = deploy_key_from_env({DEPLOY_KEY_ENV: _FAKE_KEY}, lookup=lambda _n: "from-keychain")
    assert key is not None
    assert key.source == "environment"
    assert key.private_key_pem.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")


def test_a_key_can_come_from_the_keychain():
    key = deploy_key_from_env({}, lookup=lambda name: _FAKE_KEY if name == DEPLOY_KEY_ENV else None)
    assert key is not None and key.source == "keystore"


def test_a_blank_environment_value_falls_through_to_the_keychain():
    key = deploy_key_from_env(
        {DEPLOY_KEY_ENV: "   "},
        lookup=lambda name: _FAKE_KEY if name == DEPLOY_KEY_ENV else None,
    )
    assert key is not None and key.source == "keystore"


def test_an_escaped_newline_secret_is_normalised():
    """A single-line secret store (a CI variable, a shell export) turns newlines into `\\n`."""
    escaped = _FAKE_KEY.replace("\n", "\\n").strip()
    key = deploy_key_from_env({DEPLOY_KEY_ENV: escaped})
    assert key is not None
    assert "\n" in key.private_key_pem
    assert "\\n" not in key.private_key_pem
    assert key.private_key_pem.endswith("\n")


def test_a_value_that_is_not_a_key_is_refused_rather_than_ignored():
    """A misconfigured secret must fail NOW, not an hour later inside `open_pr`."""
    with pytest.raises(CheckoutError) as caught:
        deploy_key_from_env({DEPLOY_KEY_ENV: "/home/app/.ssh/id_ed25519"})
    assert caught.value.code == "deploy_key_invalid"
    assert "path" in str(caught.value)
    # The value itself never reaches the error details.
    assert "/home/app" not in str(caught.value.details)


def test_the_key_is_never_formatted():
    key = DeployKey(private_key_pem=_FAKE_KEY, source="environment")
    for formatted in (repr(key), str(key), f"{key}"):
        assert "OPENSSH PRIVATE KEY" not in formatted
        assert "b3BlbnNzaC1rZXkt" not in formatted


# -- rewriting a remote --------------------------------------------------------------------


def test_an_https_remote_becomes_ssh():
    assert scm_url_for("https://github.com/o/r.git") == "git@github.com:o/r.git"
    # A URL with credentials in it: the userinfo is dropped, never forwarded.
    assert scm_url_for("https://x-access-token:abc@github.com/o/r") == "git@github.com:o/r"


def test_an_ssh_remote_is_unchanged():
    for remote in ("git@github.com:o/r.git", "ssh://git@github.com/o/r.git"):
        assert scm_url_for(remote) == remote


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "ftp://example.com/x", "file:///tmp/repo", "github.com/o/r", "git@github.com:o/r.git\n[core]"],
)
def test_an_unrecognised_remote_is_refused(bad: str):
    with pytest.raises(CheckoutError) as caught:
        scm_url_for(bad)
    assert caught.value.code in {"checkout_remote_missing", "checkout_remote_invalid"}


# -- the clone -----------------------------------------------------------------------------


class _Recorder:
    """A git seam that records what git WOULD be run with, and touches no network.

    It also snapshots the transient credential material, because everything in the scratch directory
    is unlinked when `checkout_repository` returns — reading it afterwards is the bug a test of
    cleanliness would hit.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, int | None]] = []
        self.envs: dict[str, Any] = {}
        self.known_hosts: str | None = None

    def __call__(self, url: str, root: Path, depth: int | None, env: dict[str, str]) -> None:
        self.calls.append((url, root, depth))
        self.envs["last"] = dict(env)
        known_hosts_path = Path(env["CODNA_KNOWN_HOSTS_PATH"])
        if known_hosts_path.is_file():
            self.known_hosts = known_hosts_path.read_text(encoding="utf-8")


def test_the_CLONE_IS_REFUSED_WITHOUT_A_KEY(tmp_path: Path):
    with pytest.raises(CheckoutError) as caught:
        checkout_repository("git@github.com:o/r.git", tmp_path / "clone")
    assert caught.value.code == "deploy_key_missing"


def test_the_key_never_appears_in_checkout_inputs_and_the_file_is_transient(tmp_path: Path):
    """The whole point: `ps` shows the wrapper, and the key is gone when the call returns."""
    recorder = _Recorder()
    destination = tmp_path / "clone"
    key = DeployKey(private_key_pem=_FAKE_KEY, source="environment")

    checkout_repository(
        "https://github.com/o/r.git", destination, deploy_key=key, git=recorder
    )

    assert recorder.calls, "git was never invoked"
    checkout_inputs = " ".join(call[0] for call in recorder.calls)
    # The remote is the SSH form, and no part of the KEY reaches the checkout runner.
    assert "git@github.com:o/r.git" in checkout_inputs
    assert "OPENSSH PRIVATE KEY" not in checkout_inputs
    assert "b3BlbnNzaC1rZXkt" not in checkout_inputs
    # `GIT_SSH_COMMAND` names a WRAPPER, not `ssh -i <key path>`, so the key path is not in argv either.
    env = recorder.envs["last"]
    assert env["GIT_SSH_COMMAND"].endswith("ssh-wrapper.sh")
    assert env["GIT_SSH_COMMAND"] not in ("ssh", "")
    # The key path travelled in the ENVIRONMENT of the wrapper, and the file is gone afterwards.
    key_path = Path(env["CODNA_DEPLOY_KEY_PATH"])
    assert not key_path.exists(), "the deploy key was left on disk after the clone"
    assert not Path(env["GIT_SSH_COMMAND"]).exists(), "the wrapper was left on disk after the clone"


def test_the_key_file_is_0600_while_it_exists(tmp_path: Path):
    """A brief world-readable window is enough to leak a key; the mode is asserted, not assumed."""
    seen: dict[str, Any] = {}

    def fake_git(_url: str, _root: Path, _depth: int | None, env: dict[str, str]) -> None:
        path = Path(env["CODNA_DEPLOY_KEY_PATH"])
        seen["mode"] = stat.S_IMODE(path.stat().st_mode)
        seen["contents"] = path.read_text(encoding="utf-8")
        seen["parent_mode"] = stat.S_IMODE(path.parent.stat().st_mode)
        seen["wrapper_mode"] = stat.S_IMODE(Path(env["GIT_SSH_COMMAND"]).stat().st_mode)
        seen["wrapper"] = Path(env["GIT_SSH_COMMAND"]).read_text(encoding="utf-8")

    checkout_repository(
        "git@github.com:o/r.git",
        tmp_path / "clone",
        deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
        git=fake_git,
    )
    assert seen["mode"] == 0o600, f"the key file is {oct(seen['mode'])}, not 0600"
    assert seen["parent_mode"] == 0o700, "the key's directory must be private (mkdtemp is 0700)"
    assert seen["wrapper_mode"] == 0o700
    assert seen["contents"].startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    # Host verification is ON, and there is no "skip" option anywhere in the wrapper.
    assert "StrictHostKeyChecking=yes" in seen["wrapper"]
    assert "UserKnownHostsFile" in seen["wrapper"]
    assert "IdentitiesOnly=yes" in seen["wrapper"]
    assert "BatchMode=yes" in seen["wrapper"]
    assert "-F /dev/null" in seen["wrapper"]
    assert "StrictHostKeyChecking=no" not in seen["wrapper"]


def test_the_key_and_the_wrapper_are_removed_on_FAILURE_too(tmp_path: Path):
    """A crash must not leave a usable credential on a shared filesystem."""
    captured: dict[str, Path] = {}

    def failing_git(_url: str, _root: Path, _depth: int | None, env: dict[str, str]) -> None:
        captured["key"] = Path(env["CODNA_DEPLOY_KEY_PATH"])
        captured["wrapper"] = Path(env["GIT_SSH_COMMAND"])
        assert captured["key"].exists()
        raise CheckoutError("clone refused", code="checkout_clone_failed")

    with pytest.raises(CheckoutError):
        checkout_repository(
            "git@github.com:o/r.git",
            tmp_path / "clone",
            deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
            git=failing_git,
        )
    assert not captured["key"].exists()
    assert not captured["wrapper"].exists()


def test_host_verification_defaults_to_github_and_cannot_be_empty(tmp_path: Path):
    recorder = _Recorder()
    checkout_repository(
        "git@github.com:o/r.git",
        tmp_path / "clone",
        deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
        git=recorder,
    )
    known_hosts = recorder.known_hosts
    assert known_hosts is not None
    assert known_hosts.rstrip("\n") == GITHUB_KNOWN_HOSTS.rstrip("\n")
    assert known_hosts.count("github.com ") == 3

    with pytest.raises(CheckoutError) as caught:
        checkout_repository(
            "git@github.com:o/r.git",
            tmp_path / "other",
            deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
            known_hosts="   ",
            git=recorder,
        )
    assert caught.value.code == "checkout_known_hosts_missing"


def test_a_non_empty_destination_is_refused_and_nothing_is_deleted(tmp_path: Path):
    destination = tmp_path / "occupied"
    destination.mkdir()
    keeper = destination / "important.txt"
    keeper.write_text("do not delete me", encoding="utf-8")

    with pytest.raises(CheckoutError) as caught:
        checkout_repository(
            "git@github.com:o/r.git",
            destination,
            deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
            git=lambda *_args: None,
        )
    assert caught.value.code == "checkout_destination_not_empty"
    assert keeper.read_text(encoding="utf-8") == "do not delete me"


def test_an_existing_EMPTY_destination_is_accepted(tmp_path: Path):
    """A mounted empty volume is the normal hosted shape, not a refusal."""
    destination = tmp_path / "empty-volume"
    destination.mkdir()
    checkout_repository(
        "git@github.com:o/r.git",
        destination,
        deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
        git=lambda *_args: None,
    )


def test_git_runs_with_a_SCRUBBED_environment(tmp_path: Path):
    """An inherited GIT_DIR or askpass would redirect auth or hang on a missing terminal."""
    seen: dict[str, Any] = {}

    def fake_git(_url: str, _root: Path, _depth: int | None, env: dict[str, str]) -> None:
        seen["env"] = dict(env)

    base = {
        "PATH": os.environ.get("PATH", ""),
        "GIT_DIR": "/somewhere/else/.git",
        "GIT_WORK_TREE": "/somewhere/else",
        "GIT_ASKPASS": "/usr/bin/false",
        "SSH_ASKPASS": "/usr/bin/false",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
        "HOME": "/home/app",
    }
    checkout_repository(
        "git@github.com:o/r.git",
        tmp_path / "clone",
        deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
        base_env=base,
        git=fake_git,
    )
    env = seen["env"]
    for dropped in ("GIT_DIR", "GIT_WORK_TREE", "GIT_ASKPASS", "SSH_ASKPASS", "SSH_AUTH_SOCK"):
        assert dropped not in env, f"{dropped} must be scrubbed: git would follow it instead of the key"
    # ...and the ordinary environment survives, so git can find its binaries and its config.
    assert env["PATH"] == base["PATH"]
    assert env["HOME"] == "/home/app"
    # Never prompt: a hosted worker has no terminal, so a prompt is a hang.
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_a_shallow_clone_is_the_default_and_can_be_made_full(tmp_path: Path):
    recorder = _Recorder()
    checkout_repository(
        "git@github.com:o/r.git",
        tmp_path / "shallow",
        deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
        git=recorder,
    )
    assert recorder.calls[0][2] == 1

    checkout_repository(
        "git@github.com:o/r.git",
        tmp_path / "full",
        deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
        depth=None,
        git=recorder,
    )
    assert recorder.calls[1][2] is None


def test_untrusted_remote_and_destination_never_reach_git_argv(tmp_path: Path, monkeypatch):
    from codna.durable import checkout as checkout_module

    destination = tmp_path / "clone with spaces;$(touch SHOULD_NOT_EXIST)"
    (destination / ".git").mkdir(parents=True)
    calls: list[tuple[list[str], dict[str, Any]]] = []
    remote = "git@github.com:o/r.git"

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(argv), kwargs))
        output = "refs/remotes/origin/trunk\n" if "symbolic-ref" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    monkeypatch.setattr(checkout_module.subprocess, "run", fake_run)
    checkout_module._run_git(remote, destination, 1, {})
    for argv, kwargs in calls:
        assert remote not in argv
        assert str(destination) not in argv
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == destination
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_clone_preserves_the_remote_default_branch_and_tracking(tmp_path: Path, monkeypatch):
    from codna.durable import checkout as checkout_module

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "probe@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Probe"], check=True)
    (source / "README.md").write_text("default branch\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "--quiet", "-m", "initial"], check=True)
    subprocess.run(["git", "-C", str(source), "branch", "-M", "trunk"], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "-b", "side"], check=True)
    (source / "README.md").write_text("side branch\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "commit", "--quiet", "-am", "side"], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "trunk"], check=True)
    monkeypatch.setattr(checkout_module, "scm_url_for", lambda _remote: str(source))

    destination = tmp_path / "clone"
    result = checkout_repository(
        "local",
        destination,
        deploy_key=DeployKey(private_key_pem=_FAKE_KEY, source="environment"),
        known_hosts="unused",
    )
    assert result == destination
    assert (destination / "README.md").read_text(encoding="utf-8") == "default branch\n"
    assert subprocess.check_output(["git", "-C", str(destination), "branch", "--show-current"], text=True).strip() == "trunk"
    assert subprocess.check_output(["git", "-C", str(destination), "remote", "get-url", "origin"], text=True).strip() == str(source)
    assert subprocess.check_output(["git", "-C", str(destination), "config", "branch.trunk.merge"], text=True).strip() == "refs/heads/trunk"
    assert subprocess.check_output(["git", "-C", str(destination), "rev-list", "--count", "HEAD"], text=True).strip() == "1"


# -- 5. `codna run --repo <remote>` is the entry point a hosted worker uses ----------------


def test_a_local_path_containing_a_colon_is_not_treated_as_a_remote():
    """`/tmp/a:b` is a directory, not a URL. The check reads schemes, not punctuation."""
    from codna.durable.cli import _looks_remote

    assert _looks_remote("/tmp/a:b") is False
    assert _looks_remote("C:/repos/x") is False
    assert _looks_remote("some/local/dir") is False
    assert _looks_remote("git@github.com:o/r.git") is True
    assert _looks_remote("ssh://git@github.com/o/r.git") is True
    assert _looks_remote("https://github.com/o/r.git") is True


def test_every_spelling_of_one_remote_clones_into_the_SAME_directory():
    """Otherwise the same repo is cloned twice and the second run edits a tree the first never saw."""
    from codna.durable.cli import _remote_directory_name

    names = {
        _remote_directory_name(remote)
        for remote in (
            "git@github.com:thyn-ai/codna.git",
            "https://github.com/thyn-ai/codna.git",
            "ssh://git@github.com/thyn-ai/codna.git",
            "https://github.com/thyn-ai/codna",
        )
    }
    assert names == {"codna"}


def test_a_remote_cannot_escape_the_clone_root_through_its_repository_name(tmp_path, monkeypatch):
    from codna.durable import cli as durable_cli

    monkeypatch.setenv("CODNA_DURABLE_REPO_ROOT", str(tmp_path / "repos"))
    with pytest.raises(Exception) as caught:
        durable_cli._clone_remote("https://github.com/thyn-ai/..")
    assert caught.value.code == "durable_repo_missing"
    assert not (tmp_path / "repos").exists()


def test_a_run_against_a_remote_clones_with_the_deploy_key(tmp_path, monkeypatch):
    """The hosted entry point: `--repo git@…` clones, then the run continues on a LOCAL path.

    Both `git` seams are injected — the clone AND the one `_clone_remote` uses to verify an existing
    clone's origin — so this asserts the WIRING (remote -> a clone under the repos root -> a local
    `repo_dir` the workflow can act on) without a network, a credential, or a real remote. The lookup
    proof is the point: a hosted worker's whole first step must not depend on a checkout existing.
    """
    from codna.durable import checkout as checkout_module
    from codna.durable import cli as durable_cli

    repos_root = tmp_path / "repos"
    monkeypatch.setenv("CODNA_DURABLE_REPO_ROOT", str(repos_root))
    monkeypatch.setenv(DEPLOY_KEY_ENV, _FAKE_KEY)

    cloned: list[tuple[str, Path, dict[str, str]]] = []

    def fake_clone(url: str, root: Path, _depth: int | None, env: dict[str, str]) -> None:
        cloned.append((url, root, dict(env)))
        # Materialise what a clone would: a working tree the run can act on.
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        (root / "README.md").write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr(checkout_module, "_run_git", fake_clone)
    resolved = durable_cli._repo_dir("git@github.com:thyn-ai/codna.git")

    # One clone, into the repos root, at the repository's NAME — not at a per-run path, so a second
    # run of the same repository reuses it instead of cloning again.
    assert resolved == (repos_root / "codna").resolve()
    assert len(cloned) == 1
    _url, _root, env = cloned[0]
    assert _url == "git@github.com:thyn-ai/codna.git"
    # The key travelled in the WRAPPER's environment, never in argv.
    assert _FAKE_KEY.splitlines()[1] not in _url
    assert env["GIT_SSH_COMMAND"].endswith("ssh-wrapper.sh")
    assert (resolved / "README.md").read_text(encoding="utf-8") == "hello\n"


def test_a_second_run_REUSES_a_verified_clone_and_never_reclones(tmp_path, monkeypatch):
    """Re-cloning on every run would discard the previous run's branch and its objects."""
    from codna.durable import checkout as checkout_module
    from codna.durable import cli as durable_cli

    repos_root = tmp_path / "repos"
    monkeypatch.setenv("CODNA_DURABLE_REPO_ROOT", str(repos_root))
    monkeypatch.setenv(DEPLOY_KEY_ENV, _FAKE_KEY)
    clones: list[Path] = []

    def fake_clone(_url: str, root: Path, _depth: int | None, _env: dict[str, str]) -> None:
        clones.append(root)
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "remote", "add", "origin",
             "git@github.com:thyn-ai/codna.git"],
            check=True,
        )

    monkeypatch.setattr(checkout_module, "_run_git", fake_clone)
    first = durable_cli._repo_dir("git@github.com:thyn-ai/codna.git")
    second = durable_cli._repo_dir("https://github.com/thyn-ai/codna.git")

    # The https spelling of the SAME repository resolves to the SAME clone, and no second clone ran:
    # `_remote_directory_name` and the origin comparison both normalise to the SSH form.
    assert first == second == (repos_root / "codna").resolve()
    assert len(clones) == 1, clones


def test_a_clone_holding_a_DIFFERENT_repository_is_refused(tmp_path, monkeypatch):
    """Reusing by NAME alone would edit someone else's checkout."""
    from codna.durable import cli as durable_cli

    repos_root = tmp_path / "repos"
    occupancy = repos_root / "codna"
    occupancy.mkdir(parents=True)
    (occupancy / "file.txt").write_text("someone else", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(occupancy)], check=True)
    subprocess.run(
        ["git", "-C", str(occupancy), "remote", "add", "origin", "git@github.com:other/project.git"],
        check=True,
    )
    monkeypatch.setenv("CODNA_DURABLE_REPO_ROOT", str(repos_root))

    with pytest.raises(Exception) as caught:
        durable_cli._repo_dir("git@github.com:thyn-ai/codna.git")
    assert getattr(caught.value, "code", "") == "durable_repo_conflict"


def test_a_remote_with_no_deploy_key_is_refused_by_name(tmp_path, monkeypatch):
    """A hosted run that cannot authenticate fails at the START, naming the variable, not in open_pr."""
    from codna.durable import checkout as checkout_module
    from codna.durable import cli as durable_cli

    monkeypatch.setenv("CODNA_DURABLE_REPO_ROOT", str(tmp_path / "repos"))
    monkeypatch.delenv(DEPLOY_KEY_ENV, raising=False)
    monkeypatch.setattr(checkout_module, "deploy_key_from_env", lambda env=None, lookup=None: None)

    with pytest.raises(Exception) as caught:
        durable_cli._repo_dir("git@github.com:thyn-ai/codna.git")
    assert getattr(caught.value, "code", "") == "deploy_key_missing"
