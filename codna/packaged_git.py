from __future__ import annotations

import os
import re
import subprocess
from .patch_text import GIT_APPLY_FLAG_SETS, normalize_unified_diff
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx


GIT_TIMEOUT_SECONDS = 120
# The initial clone is the one git step that scales with repository size. A FULL clone of
# thyn-ai/algenta (~81 MB packed) hit the flat 120 s bound every time on the 2-vCPU webhook
# machine (2026-09-17, `@codna fix` -> "git clone ... timed out after 120 seconds"), so the
# clone is shallow and gets its own bound.
GIT_CLONE_TIMEOUT_SECONDS = 300
GITHUB_HTTP_TIMEOUT_SECONDS = 60
_GITHUB_URL_RE = re.compile(r"^https://github\.com/([^/]+)/([^/.]+)(?:\.git)?/?$")
_SSH_GITHUB_URL_RE = re.compile(r"^git@github\.com:([^/]+)/([^/.]+)(?:\.git)?$")


class PackagedGitError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class MaterializedRepository:
    path: Path
    repository_url: str | None
    repository_slug: str | None
    access_token: str | None


def materialize_repository(
    *,
    runtime_root: Path,
    connector_id: str,
    connector_type: str,
    config: Mapping[str, Any],
    request: Mapping[str, Any],
) -> MaterializedRepository:
    if connector_type == "local_repo":
        return _local_repository(config)
    if connector_type == "github_repo":
        return _github_repository(
            runtime_root=runtime_root,
            connector_id=connector_id,
            config=config,
            request=request,
        )
    raise PackagedGitError(
        "local_repository_packaged_backend_unsupported_connector",
        "The packaged local backend supports local_repo and github_repo connectors.",
        {"connector_type": connector_type},
    )


def open_remote_pr(
    *,
    repo_root: Path,
    repository_url: str,
    repository_slug: str | None,
    access_token: str | None,
    plan_id: str,
    patch_diff: str,
    base_branch: str | None,
    title: str,
    body: str,
) -> dict[str, str]:
    token = _required_token(access_token)
    base = _validated_branch(base_branch or default_branch(repo_root))
    branch = _validated_branch("codna/" + plan_id.removeprefix("plan_")[:16])
    slug = repository_slug or github_slug_from_url(repository_url)
    if not slug:
        raise PackagedGitError(
            "repository_remote_pr_slug_required",
            "remote_pr apply requires a GitHub repository URL or repository_slug.",
            {"repository_url": _redact_url(repository_url)},
        )

    _require_git_repo(repo_root)
    _reset_owned_checkout(repo_root)
    _run_git(repo_root, ["checkout", "-B", branch, "HEAD"])
    _apply_patch_to_index(repo_root, patch_diff)
    _run_git(
        repo_root,
        [*_git_identity_args(), "commit", "-m", title],
    )
    try:
        _run_git(
            repo_root,
            ["push", "origin", f"HEAD:refs/heads/{branch}"],
            env=_git_auth_env(repository_url, token),
        )
    except PackagedGitError as exc:
        stderr = str((exc.details or {}).get("stderr", ""))
        if "without `workflows` permission" in stderr:
            # GitHub compares a NEW branch's .github/workflows against the default branch, so an
            # App token without `workflows` cannot push a fix onto any PR branch that predates a
            # workflow change on main -- even when the fix touches no workflow at all.
            raise PackagedGitError(
                "workflows_permission_required",
                "GitHub refused the push: this branch's .github/workflows files differ from the default "
                "branch and the Codna App does not have the `workflows` permission. Grant it in the App's "
                "settings (Repository permissions -> Workflows: Read and write) and accept it on the "
                "installation, or update the PR branch from the default branch, then re-run `@codna fix`.",
                {"branch": branch, "stderr": stderr[-800:]},
            ) from exc
        raise
    pull_request_url = create_github_pull_request(
        repository_slug=slug,
        token=token,
        branch=branch,
        base_branch=base,
        title=title,
        body=body,
    )
    return {
        "branch_name": branch,
        "base_branch": base,
        "repository_slug": slug,
        "pull_request_url": pull_request_url,
    }


def create_github_pull_request(
    *,
    repository_slug: str,
    token: str,
    branch: str,
    base_branch: str,
    title: str,
    body: str,
) -> str:
    headers = {"authorization": f"Bearer {token}", "accept": "application/vnd.github+json"}
    response = httpx.post(
        f"https://api.github.com/repos/{repository_slug}/pulls",
        headers=headers,
        timeout=GITHUB_HTTP_TIMEOUT_SECONDS,
        json={"title": title, "head": branch, "base": base_branch, "body": body},
    )
    if response.status_code >= 400:
        raise PackagedGitError(
            "repository_remote_pr_create_failed",
            "GitHub rejected the pull request creation request.",
            {"repository_slug": repository_slug, "status_code": response.status_code, "body": response.text[:500]},
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise PackagedGitError(
            "repository_remote_pr_create_failed",
            "GitHub returned a non-JSON pull request response.",
            {"repository_slug": repository_slug, "status_code": response.status_code},
        ) from exc
    url = payload.get("html_url")
    if not isinstance(url, str) or not url:
        raise PackagedGitError(
            "repository_remote_pr_create_failed",
            "GitHub pull request response did not include html_url.",
            {"repository_slug": repository_slug, "status_code": response.status_code},
        )
    return url


def default_branch(repo_root: Path) -> str:
    result = _run_git(repo_root, ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], check=False)
    if result.returncode == 0:
        value = result.stdout.strip()
        if value.startswith("origin/"):
            return value.removeprefix("origin/")
    return "main"


def github_slug_from_url(repository_url: str) -> str | None:
    for pattern in (_GITHUB_URL_RE, _SSH_GITHUB_URL_RE):
        match = pattern.match(repository_url.strip())
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    return None


def _local_repository(config: Mapping[str, Any]) -> MaterializedRepository:
    raw_path = config.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise PackagedGitError(
            "invalid_connector_config",
            "Local repository connector config requires a non-empty path.",
            {"connector_type": "local_repo"},
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise PackagedGitError(
            "invalid_connector_config",
            "Local repository connector path must exist and be a directory.",
            {"path": str(path)},
        )
    return MaterializedRepository(path=path, repository_url=None, repository_slug=None, access_token=None)


def _github_repository(
    *,
    runtime_root: Path,
    connector_id: str,
    config: Mapping[str, Any],
    request: Mapping[str, Any],
) -> MaterializedRepository:
    repository_url = _required_config_string(config, "repository_url")
    _reject_embedded_credentials(repository_url)
    access_token = _optional_config_string(config, "access_token")
    repository_slug = _optional_config_string(config, "repository_slug") or github_slug_from_url(repository_url)
    checkout = runtime_root / "repository-intelligence" / "packaged" / "checkouts" / connector_id
    if (checkout / ".git").is_dir():
        _reset_owned_checkout(checkout)
        _run_git(checkout, ["remote", "set-url", "origin", repository_url])
        _run_git(
            checkout, ["fetch", "--prune", "--depth=1", "origin"],
            env=_git_auth_env(repository_url, access_token), timeout=GIT_CLONE_TIMEOUT_SECONDS,
        )
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        # Shallow: _checkout_requested_ref fetches the exact requested ref at depth 1 right after,
        # so history beyond the tip was downloaded and never used.
        _run_git(
            None,
            ["clone", "--no-tags", "--depth=1", repository_url, str(checkout)],
            env=_git_auth_env(repository_url, access_token),
            timeout=GIT_CLONE_TIMEOUT_SECONDS,
        )
    _checkout_requested_ref(checkout, request, repository_url=repository_url, access_token=access_token)
    return MaterializedRepository(
        path=checkout.resolve(),
        repository_url=repository_url,
        repository_slug=repository_slug,
        access_token=access_token,
    )


def _checkout_requested_ref(
    repo_root: Path,
    request: Mapping[str, Any],
    *,
    repository_url: str,
    access_token: str | None,
) -> None:
    ref = request.get("ref")
    if isinstance(ref, str) and ref.strip():
        clean_ref = _validated_ref(ref)  # reject option-like / malformed refs (arg-injection RCE)
        fetch = _run_git(
            repo_root,
            ["fetch", "--depth=1", "origin", clean_ref],
            env=_git_auth_env(repository_url, access_token),
            check=False,
        )
        # On a clean fetch use the literal FETCH_HEAD; otherwise fall back to the validated ref
        # (guaranteed not to begin with '-', so git can never read it as an option).
        target = "FETCH_HEAD" if fetch.returncode == 0 else clean_ref
        _run_git(repo_root, ["checkout", "--detach", target])
    else:
        branch = default_branch(repo_root)
        _run_git(repo_root, ["checkout", "-B", branch, f"origin/{branch}"])
    _reset_owned_checkout(repo_root)


def _apply_patch_to_index(repo_root: Path, patch_diff: str) -> None:
    patch = normalize_unified_diff(patch_diff)
    if not patch.strip():
        raise PackagedGitError("repository_remote_pr_empty_patch", "remote_pr apply requires a non-empty patch.", {})
    first_error = ""
    for flags in GIT_APPLY_FLAG_SETS:
        check = _run_git_stdin(repo_root, ["apply", "--check", "--whitespace=nowarn", *flags, "-"], patch, check=False)
        if check.returncode == 0:
            _run_git_stdin(repo_root, ["apply", "--index", "--whitespace=nowarn", *flags, "-"], patch)
            return
        first_error = first_error or check.stderr.strip()
    raise PackagedGitError(
        "patch_rejected",
        "The generated patch does not apply to the checkout: "
        + (first_error.splitlines()[0] if first_error else "git apply --check failed"),
        {"stderr": first_error[-2000:]},
    )


def _require_git_repo(repo_root: Path) -> None:
    result = _run_git(repo_root, ["rev-parse", "--is-inside-work-tree"], check=False)
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise PackagedGitError(
            "repository_apply_requires_git",
            "remote_pr apply requires a git repository.",
            {"repo_root": str(repo_root)},
        )


def _reset_owned_checkout(repo_root: Path) -> None:
    _run_git(repo_root, ["reset", "--hard", "HEAD"])
    _run_git(repo_root, ["clean", "-fdx"])


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _validated_ref(ref: str) -> str:
    """Validate a user/webhook-supplied git ref before it reaches a git positional.

    Closes an argument-injection RCE: an unvalidated ref that begins with ``-`` (e.g.
    ``--upload-pack=<cmd>``) is parsed by git as an option and can execute arbitrary
    commands. A ref must be either a hex commit SHA or a well-formed refname, and must
    never begin with ``-``.
    """
    clean = ref.strip()
    if not clean or clean != ref or clean.startswith("-"):
        raise PackagedGitError("repository_invalid_ref", "Requested git ref is invalid.", {"ref": ref})
    if 7 <= len(clean) <= 64 and all(c in _HEX_DIGITS for c in clean):
        return clean  # commit SHA — safe positional, cannot be an option
    result = subprocess.run(
        ["git", "check-ref-format", "--allow-onelevel", clean],
        capture_output=True, text=True, check=False, timeout=10,
    )
    if result.returncode != 0:
        raise PackagedGitError("repository_invalid_ref", "Requested git ref is invalid.", {"ref": ref})
    return clean


def _validated_branch(branch: str) -> str:
    if not branch or branch.strip() != branch:
        raise PackagedGitError("repository_remote_pr_invalid_branch", "Git branch name is invalid.", {"branch": branch})
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise PackagedGitError("repository_remote_pr_invalid_branch", "Git branch name is invalid.", {"branch": branch})
    return branch


def _required_token(value: str | None) -> str:
    if not value:
        raise PackagedGitError(
            "repository_remote_pr_token_required",
            "remote_pr apply requires a write token.",
            {},
        )
    return value


def _required_config_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PackagedGitError(
            "invalid_connector_config",
            f"GitHub repository connector config requires a non-empty {key}.",
            {"field": key},
        )
    return value.strip()


def _optional_config_string(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _reject_embedded_credentials(repository_url: str) -> None:
    if "://" not in repository_url:
        return
    parsed = urlsplit(repository_url)
    if parsed.username or parsed.password:
        raise PackagedGitError(
            "invalid_connector_config",
            "Repository URL must not embed credentials; pass the token separately.",
            {"field": "repository_url"},
        )


def _git_auth_env(repository_url: str, token: str | None) -> dict[str, str] | None:
    env = dict(os.environ)
    # Hermetic: never inherit ambient GIT_CONFIG_* injection from the caller's shell.
    # codna's git operations must use only the config codna sets here — a security product
    # cannot let the surrounding environment silently alter how it clones or pushes.
    for key in [k for k in env if k == "GIT_CONFIG_COUNT" or k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))]:
        env.pop(key, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token and repository_url.startswith(("http://", "https://")):
        credential = b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {credential}"
    return env


# The webhook worker sets these to the GitHub App's bot identity ("<slug>[bot]",
# "<id>+<slug>[bot]@users.noreply.github.com") so fix commits resolve to a real GitHub account.
# A CLA bot on thyn-ai/algenta-integrations#80 (2026-09-17) rejected "codna <codna@codna.ai>"
# with "codna seems not to be a GitHub user"; the bot login is what CLA allowlists can name.
DEFAULT_GIT_USER_NAME = "codna"
DEFAULT_GIT_USER_EMAIL = "codna@codna.ai"


def _git_identity_args() -> list[str]:
    name = os.environ.get("CODNA_GIT_USER_NAME") or DEFAULT_GIT_USER_NAME
    email = os.environ.get("CODNA_GIT_USER_EMAIL") or DEFAULT_GIT_USER_EMAIL
    return ["-c", f"user.email={email}", "-c", f"user.name={name}"]


def _run_git(
    cwd: Path | None,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS if timeout is None else timeout,
    )
    if check and result.returncode != 0:
        raise PackagedGitError(
            "git_command_failed",
            "Git command failed while preparing packaged repository access.",
            {"args": _redact_args(args), "cwd": str(cwd) if cwd else None, "stderr": result.stderr[-2000:]},
        )
    return result


def _run_git_stdin(
    repo_root: Path, args: list[str], stdin: str, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if check and result.returncode != 0:
        raise PackagedGitError(
            "git_command_failed",
            "Git command failed while applying the packaged local patch.",
            {"args": _redact_args(args), "cwd": str(repo_root), "stderr": result.stderr[-2000:]},
        )
    return result

def _redact_args(args: list[str]) -> list[str]:
    return [_redact_url(value) for value in args]


def _redact_url(value: str) -> str:
    if "@" not in value or "://" not in value:
        return value
    scheme, rest = value.split("://", 1)
    return f"{scheme}://<redacted>@{rest.rsplit('@', 1)[-1]}"
