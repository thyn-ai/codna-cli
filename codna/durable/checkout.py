"""HOW a hosted worker gets repository access (D6): a deploy key, resolved at RUN time, never baked.

A hosted worker has to clone and push to a repository it does not own. The three ways to do that are
not equivalent:

* **A long-lived token in the image** — refused. An image is a published artifact: every registry
  copy, every `docker history`, every builder cache holds it. There is no version of this that is
  safe, so this module cannot express it.
* **A token in an environment variable at run time** — works, and is what `github_token_provider`
  already does for the API. But it is a broad credential: a PAT can read everything its owner can,
  and a worker that only needs one repository holds a key to all of them.
* **A deploy key** (an SSH key granted to ONE repository, read/write or read-only) — the narrowest
  credential that does the job, and the one this module implements. It arrives through the platform
  secret store (Fly secrets, a k8s Secret) or the developer keychain, is materialised into a
  `0600` file for the duration of one git operation, and is removed afterwards.

## What is guaranteed, and what is not

Guaranteed by this module:

* the key never appears in a command line (`git` is told to use a *wrapper script path*, and the
  key path is inside that wrapper's environment — `ps` shows neither the key nor the wrapper's
  contents);
* the key file is `0600`, inside a private temporary directory, and unlinked in a `finally`, so a
  crash cannot leave a usable credential on a shared filesystem;
* host verification is ON and defaults to GitHub's published keys, so a man-in-the-middle cannot
  collect the key by presenting its own host key. An operator who uses a self-hosted remote names
  its host keys explicitly (`CODNA_DURABLE_GIT_KNOWN_HOSTS`); there is no "skip verification" mode.

NOT guaranteed: an operator with root on the machine can read the key file while it exists. A
deploy key scoped to one repository is the mitigation, not the absence of the risk — and it is why
the key must be a deploy key rather than a PAT.

`git` runs with a SCRUBBED environment: the variables in `GIT_ENV_DROP` would otherwise redirect it
at an unrelated repository, and an inherited `GIT_ASKPASS`/credential helper could make it ignore the
key entirely (or prompt a non-existent terminal and hang).
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from .worktree import GIT_ENV_DROP

__all__ = [
    "DEPLOY_KEY_ENV",
    "GITHUB_KNOWN_HOSTS",
    "KNOWN_HOSTS_ENV",
    "CheckoutError",
    "DeployKey",
    "checkout_repository",
    "deploy_key_from_env",
    "looks_remote",
    "scm_url_for",
]

#: The deploy key, as PEM text. A PLATFORM secret (`fly secrets set`), never an image ENV. Named
#: without a trailing `_PATH` because it holds the KEY ITSELF: pointing it at a file would make the
#: value a path that has to exist in the image, which is the thing this module forbids.
DEPLOY_KEY_ENV: Final = "CODNA_DURABLE_DEPLOY_KEY"
#: Optional: known-hosts text for a self-hosted remote. Absent means GitHub only.
KNOWN_HOSTS_ENV: Final = "CODNA_DURABLE_GIT_KNOWN_HOSTS"

# GitHub's published SSH host keys (api.github.com/meta -> ssh_keys), pinned here so a hosted worker
# verifies the host by default. If GitHub ever rotates these the failure is a loud "host key
# verification failed", which is the correct failure: silently trusting a new key is the attack.
GITHUB_KNOWN_HOSTS: Final = (
    "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n"
    "github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=\n"
    "github.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1gvwnGLVlOhGeYrnZaMgRK6+PKCUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/hWCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JHnsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7GidjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA179cnfGWOWRVruj16z6XyvxvjJwbz0wQZ75XK5tKSb7FNyeIEs4TT4jk+S4dhPeAUC5y+bDYirYgM4GC7uEnztnZyaVWQ7B381AK4Qdrwt51ZqExKbQpTUNn+EjqoTwvqNj4kqx5QUCI0ThS/YkOxJCXmPUWZbhjpCg56i+2aB6CmK2JGhn57K5mj0MNdBXA4/WnwH6XoPWJzK5Nyu2zB3nAZp+S5hpQs+p1vN1/wsjk=\n"
)

_GIT_TIMEOUT_S: Final = 900
# Both directions: a worker must clone AND push, or `open_pr` cannot publish the branch.
_REQUIRED_PREFIXES: Final = ("git@", "ssh://")


class CheckoutError(RuntimeError):
    """A hosted checkout could not be prepared. Always carries a stable ``code``."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


@dataclass(frozen=True)
class DeployKey:
    """One resolved deploy key: its PEM text and where it came from. Redacted when formatted.

    ``source`` ("environment" or "keystore") is provenance an operator needs and a value map would
    leak; it is the only thing this type reports about where the key came from.
    """

    private_key_pem: str
    source: str

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return f"DeployKey(source={self.source!r}, bytes={len(self.private_key_pem)})"

    __str__ = __repr__


def _lookup_default(name: str) -> str | None:
    """Read a value from the developer keychain, or None. Never raises."""
    try:
        from ..keystore import get_key
    except Exception:  # noqa: BLE001 - no keystore module means "environment only"
        return None
    try:
        return get_key(name)
    except Exception:  # noqa: BLE001 - an unreadable keychain is "absent"
        return None


def deploy_key_from_env(
    env: Mapping[str, str] | None = None,
    *,
    lookup: Callable[[str], str | None] | None = None,
) -> DeployKey | None:
    """The deploy key this process may use, or None when it was configured with a token instead.

    ``None`` is not a failure: `github_token_provider` already covers the HTTPS-token path, and a
    development host has no deploy key at all. What IS a failure is a value that is present but is
    not a key — that is a misconfigured secret, and silently ignoring it would make a hosted worker
    fall through to "no credential" and fail an hour later inside `open_pr`.
    """
    source = os.environ if env is None else env
    raw = source.get(DEPLOY_KEY_ENV)
    origin = "environment"
    if not (isinstance(raw, str) and raw.strip()):
        reader = lookup if lookup is not None else _lookup_default
        raw = reader(DEPLOY_KEY_ENV)
        origin = "keystore"
    if not (isinstance(raw, str) and raw.strip()):
        return None
    text = raw.strip()
    if "-----BEGIN" not in text or "PRIVATE KEY-----" not in text:
        raise CheckoutError(
            f"{DEPLOY_KEY_ENV} does not contain an SSH private key (no -----BEGIN ... PRIVATE KEY----- "
            "line). Store the key FILE'S CONTENTS as the secret, not its path.",
            code="deploy_key_invalid",
            details={"variable": DEPLOY_KEY_ENV, "received_chars": len(text), "source": origin},
        )
    # `\n`-escaped values are what a single-line secret store (a CI variable, a shell export) produces
    # for a multi-line PEM. Converting here rather than at the file write keeps one normalisation.
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return DeployKey(private_key_pem=text if text.endswith("\n") else f"{text}\n", source=origin)


def looks_remote(raw: str) -> bool:
    """Whether `--repo` names a REMOTE repository rather than a local directory.

    A local path that happens to contain a colon is still a path (`/tmp/a:b`), so the check reads the
    schemes (`git@`, `ssh://`, `https://`, `http://`) rather than merely looking for `:`.
    """
    value = raw.strip()
    return value.startswith(("git@", "ssh://", "https://", "http://"))


def scm_url_for(remote: str) -> str:
    """An HTTPS/`git@` remote expressed as the SSH URL a deploy key authenticates.

    A deploy key is an SSH credential, so an `https://github.com/o/r.git` remote has to be expressed
    as `git@github.com:o/r.git` before git will offer the key. Anything already SSH-shaped is
    returned unchanged, and anything else is REFUSED rather than guessed: rewriting a remote this
    module does not recognise could push a branch to a host nobody intended.
    """
    value = (remote or "").strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise CheckoutError(
            "a hosted checkout remote cannot contain control characters.",
            code="checkout_remote_invalid",
            details={"remote": repr(value)},
        )
    if not value:
        raise CheckoutError(
            "a hosted checkout needs a remote URL.",
            code="checkout_remote_missing",
            details={"remote": ""},
        )
    if value.startswith(_REQUIRED_PREFIXES) or (
        # scp-like `user@host:path`, which has no scheme
        "@" in value.split(":", 1)[0] and "://" not in value
    ):
        return value
    if value.startswith("https://"):
        host_path = value[len("https://") :]
        if "@" in host_path:
            host_path = host_path.split("@", 1)[1]
        host, _, path = host_path.partition("/")
        if not host or not path:
            raise CheckoutError(
                f"cannot derive an SSH remote from {value!r}.",
                code="checkout_remote_invalid",
                details={"remote": value},
            )
        return f"git@{host}:{path}"
    raise CheckoutError(
        f"a hosted checkout needs an SSH or https remote; got {value!r}.",
        code="checkout_remote_invalid",
        details={"remote": value, "allowed": ["git@host:path", "ssh://...", "https://..."]},
    )


def _git_env(
    *,
    key_path: Path,
    known_hosts_path: Path,
    wrapper_path: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment one git command runs with: the key's location, nothing that redirects git.

    `GIT_SSH_COMMAND` names a WRAPPER SCRIPT rather than `ssh -i <path>` for two reasons: the key
    path then does not appear in `ps` output, and the option set (identity, batch mode, known hosts,
    no agent) lives in one auditable place instead of being re-quoted at every call site.
    """
    source = dict(os.environ if base is None else base)
    env = {key: value for key, value in source.items() if key not in GIT_ENV_DROP}
    # A credential helper or askpass inherited from the image would answer for git instead of the key,
    # or block waiting on a terminal that does not exist. Neither is permitted here.
    for name in (
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "SSH_AUTH_SOCK",
    ):
        env.pop(name, None)
    env.update(
        {
            "GIT_SSH_COMMAND": str(wrapper_path),
            # Never let git prompt: a hosted worker has no terminal, and a prompt is a hang.
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "CODNA_DEPLOY_KEY_PATH": str(key_path),
            "CODNA_KNOWN_HOSTS_PATH": str(known_hosts_path),
        }
    )
    return env


def _write_private(path: Path, text: str) -> None:
    """Write `0600` via an exclusive create, so the key is never briefly world-readable."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The descriptor is closed by `fdopen`'s context; the FILE must not survive a failed write.
        path.unlink(missing_ok=True)
        raise


def _wrapper_script() -> str:
    """The `GIT_SSH_COMMAND` wrapper: `ssh` with the key, strict host checking, no agent.

    `-F /dev/null` ignores every ssh config the image might carry, so the connection is exactly what
    is written here. `IdentitiesOnly=yes` stops ssh from also offering agent keys or `~/.ssh/id_*`
    that happen to exist in the container.
    """
    return (
        "#!/bin/sh\n"
        'exec ssh -F /dev/null -o IdentitiesOnly=yes -o BatchMode=yes \\\n'
        '  -o StrictHostKeyChecking=yes \\\n'
        '  -o UserKnownHostsFile="$CODNA_KNOWN_HOSTS_PATH" \\\n'
        '  -i "$CODNA_DEPLOY_KEY_PATH" "$@"\n'
    )


def checkout_repository(
    remote: str,
    destination: Path,
    *,
    deploy_key: DeployKey | None = None,
    known_hosts: str | None = None,
    base_env: Mapping[str, str] | None = None,
    git: Callable[..., Any] | None = None,
    depth: int | None = 1,
) -> Path:
    """Clone ``remote`` into ``destination`` with ``deploy_key``, or refuse and say why.

    ``git`` is injectable (it receives ``(url, destination, depth, env)``) so a test can assert the
    checkout inputs and environment without a network or a key. The real runner keeps untrusted
    values out of Git's argv.

    An EMPTY destination is required but MAY exist (a mounted empty directory is normal); a
    destination with anything in it is refused, never cleaned: deleting files inside a path a
    caller handed us is how a `--force` mistake destroys a checkout.
    """
    url = scm_url_for(remote)
    root = Path(destination).expanduser()
    if root.exists() and any(root.iterdir()):
        raise CheckoutError(
            f"refusing to clone into {root}: it is not empty.",
            code="checkout_destination_not_empty",
            details={"destination": str(root)},
        )
    root.mkdir(parents=True, exist_ok=True)
    hosts_text = known_hosts if known_hosts is not None else (
        (base_env or os.environ).get(KNOWN_HOSTS_ENV) or GITHUB_KNOWN_HOSTS
    )
    if not hosts_text.strip():
        raise CheckoutError(
            f"{KNOWN_HOSTS_ENV} is empty; host verification cannot be disabled.",
            code="checkout_known_hosts_missing",
            details={"variable": KNOWN_HOSTS_ENV},
        )

    # A private directory for the credential material: `mkdtemp` is 0700 by contract, so the key and
    # the wrapper are readable by this user alone even before their own modes are applied.
    with tempfile.TemporaryDirectory(prefix="codna-deploy-key-") as scratch:
        secret_dir = Path(scratch)
        key_path = secret_dir / "id_deploy"
        wrapper_path = secret_dir / "ssh-wrapper.sh"
        known_hosts_path = secret_dir / "known_hosts"
        if deploy_key is None:
            raise CheckoutError(
                "a hosted checkout needs a deploy key (or the token path, which does not come "
                "through this function).",
                code="deploy_key_missing",
                details={"variable": DEPLOY_KEY_ENV},
            )
        _write_private(key_path, deploy_key.private_key_pem)
        _write_private(wrapper_path, _wrapper_script())
        wrapper_path.chmod(0o700)
        _write_private(known_hosts_path, hosts_text if hosts_text.endswith("\n") else f"{hosts_text}\n")
        env = _git_env(
            key_path=key_path,
            known_hosts_path=known_hosts_path,
            wrapper_path=wrapper_path,
            base=base_env,
        )
        runner = git if git is not None else _run_git
        runner(url, root, depth, env)
        # `TemporaryDirectory.__exit__` removes the key, the wrapper and the known-hosts file. The
        # clone itself keeps no reference to any of them: git copies nothing out of `GIT_SSH_COMMAND`.
    return root


def _run_git(url: str, root: Path, depth: int | None, env: Mapping[str, str]) -> None:
    """Clone without putting HTTP-controlled values in a command argument."""
    fetch_env = dict(env)
    fetch_env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "remote.origin.url",
            "GIT_CONFIG_VALUE_0": url,
        }
    )
    _run_git_command(["git", "init", "--quiet"], root, env)
    fetch = ["git", "fetch", "--quiet"]
    if depth is not None:
        fetch.extend(["--depth", str(int(depth))])
    fetch.extend(["origin", "+refs/heads/*:refs/remotes/origin/*"])
    _run_git_command(fetch, root, fetch_env)
    _run_git_command(["git", "remote", "set-head", "origin", "--auto"], root, fetch_env)
    completed = _run_git_command(["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], root, fetch_env)
    branch = _local_branch(completed.stdout.strip())
    _run_git_command(
        ["git", "update-ref", "--stdin"],
        root,
        env,
        input_text=f"create refs/heads/{branch} refs/remotes/origin/{branch}\n",
    )
    _write_checkout_config(root, url, branch)
    _run_git_command(["git", "checkout", "--quiet"], root, env)


def _run_git_command(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            env=dict(env),
            check=False,
            shell=False,
            cwd=cwd,
            input=input_text,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckoutError(
            f"git could not run: {exc}",
            code="checkout_git_unavailable",
            details={"operation": argv[1], "error": type(exc).__name__},
        ) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()[:600]
        raise CheckoutError(
            f"git {argv[1]} exited {completed.returncode}: {stderr}",
            code="checkout_clone_failed",
            details={"operation": argv[1], "returncode": completed.returncode},
        )
    return completed


def _local_branch(remote_head: str) -> str:
    prefix = "refs/remotes/origin/"
    branch = remote_head.removeprefix(prefix)
    if (
        remote_head == branch
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch)
        or ".." in branch
        or "@{" in branch
        or "//" in branch
        or branch.endswith((".", "/"))
    ):
        raise CheckoutError(
            f"the remote's default HEAD is not a safe branch ref: {remote_head!r}",
            code="checkout_head_invalid",
            details={"remote_head": remote_head},
        )
    return branch


def _write_checkout_config(root: Path, url: str, branch: str) -> None:
    quoted_url = url.replace("\\", "\\\\").replace('"', '\\"')
    (root / ".git" / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    with (root / ".git" / "config").open("a", encoding="utf-8") as config:
        config.write(
            f'\n[remote "origin"]\n\turl = "{quoted_url}"\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
            f'[branch "{branch}"]\n\tremote = origin\n\tmerge = refs/heads/{branch}\n'
        )


def describe_for_command(env: Mapping[str, str] | None = None) -> str:
    """A one-line, secret-free description for a deploy preflight or a startup banner."""
    key = deploy_key_from_env(env)
    if key is None:
        return "no deploy key configured (the HTTPS token path applies)"
    return f"deploy key from {key.source} ({len(key.private_key_pem)} bytes)"
