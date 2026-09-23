"""The packaged GitHub checkout must be shallow and get the clone-sized bound (2026-09-17: a
full clone of an 81 MB repo hit the flat 120 s git timeout on every `@codna fix`)."""
from __future__ import annotations

import subprocess

from codna import packaged_git


def _capture(monkeypatch):
    calls = []

    def fake_run_git(cwd, args, *, env=None, check=True, timeout=None):
        calls.append({"cwd": cwd, "args": list(args), "timeout": timeout})
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    monkeypatch.setattr(packaged_git, "_run_git", fake_run_git)
    monkeypatch.setattr(packaged_git, "_reset_owned_checkout", lambda repo_root: None)
    return calls


def test_fresh_checkout_is_a_shallow_clone_with_the_clone_bound(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    packaged_git._github_repository(
        runtime_root=tmp_path, connector_id="c1",
        config={"repository_url": "https://github.com/acme/app.git"}, request={"ref": "feature"},
    )
    clone = calls[0]
    assert clone["args"][:3] == ["clone", "--no-tags", "--depth=1"], clone
    assert clone["args"][3] == "https://github.com/acme/app.git"
    assert clone["timeout"] == packaged_git.GIT_CLONE_TIMEOUT_SECONDS
    # the requested ref is still fetched shallowly and checked out afterwards
    assert any(c["args"][:3] == ["fetch", "--depth=1", "origin"] for c in calls[1:])


def test_existing_checkout_refreshes_shallowly(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    (tmp_path / "repository-intelligence" / "packaged" / "checkouts" / "c1" / ".git").mkdir(parents=True)
    packaged_git._github_repository(
        runtime_root=tmp_path, connector_id="c1",
        config={"repository_url": "https://github.com/acme/app.git"}, request={"ref": "feature"},
    )
    fetch = next(c for c in calls if c["args"][0] == "fetch" and "--prune" in c["args"])
    assert "--depth=1" in fetch["args"] and fetch["timeout"] == packaged_git.GIT_CLONE_TIMEOUT_SECONDS
    assert not any(c["args"][0] == "clone" for c in calls)



def test_push_refused_for_missing_workflows_permission_is_reported_actionably(monkeypatch, tmp_path):
    from codna import packaged_git as pg

    def fake_run_git(cwd, args, *, env=None, check=True, timeout=None):
        if args[0] == "push":
            raise pg.PackagedGitError("git_command_failed", "Git command failed while preparing packaged repository access.",
                                      {"args": args, "stderr": "To https://github.com/acme/app.git\n ! [remote rejected] HEAD -> codna/abc "
                                       "(refusing to allow a GitHub App to create or update workflow `.github/workflows/security.yml` "
                                       "without `workflows` permission)\nerror: failed to push some refs\n"})
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    monkeypatch.setattr(pg, "_run_git", fake_run_git)
    monkeypatch.setattr(pg, "_require_git_repo", lambda repo_root: None)
    monkeypatch.setattr(pg, "_reset_owned_checkout", lambda repo_root: None)
    monkeypatch.setattr(pg, "_apply_patch_to_index", lambda repo_root, patch: None)
    try:
        pg.open_remote_pr(repo_root=tmp_path, repository_url="https://github.com/acme/app.git", repository_slug="acme/app",
                          access_token="tok", plan_id="plan_abc123", patch_diff="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
                          base_branch="main", title="t", body="b")
    except pg.PackagedGitError as exc:
        assert exc.code == "workflows_permission_required"
        assert "`workflows` permission" in str(exc) and "update the PR branch" in str(exc)
    else:
        raise AssertionError("expected PackagedGitError")



def test_remote_pr_commits_with_the_identity_from_the_environment(monkeypatch, tmp_path):
    from codna import packaged_git as pg

    calls = []

    def fake_run_git(cwd, args, *, env=None, check=True, timeout=None):
        calls.append(list(args))
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    monkeypatch.setattr(pg, "_run_git", fake_run_git)
    monkeypatch.setattr(pg, "_require_git_repo", lambda repo_root: None)
    monkeypatch.setattr(pg, "_reset_owned_checkout", lambda repo_root: None)
    monkeypatch.setattr(pg, "_apply_patch_to_index", lambda repo_root, patch: None)
    monkeypatch.setattr(pg, "create_github_pull_request", lambda **kw: "https://github.com/acme/app/pull/1")
    monkeypatch.setenv("CODNA_GIT_USER_NAME", "codna-ai[bot]")
    monkeypatch.setenv("CODNA_GIT_USER_EMAIL", "293953567+codna-ai[bot]@users.noreply.github.com")
    pg.open_remote_pr(repo_root=tmp_path, repository_url="https://github.com/acme/app.git", repository_slug="acme/app",
                      access_token="tok", plan_id="plan_abc123", patch_diff="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
                      base_branch="main", title="t", body="b")
    commit = next(c for c in calls if "commit" in c)
    assert "user.name=codna-ai[bot]" in commit and "user.email=293953567+codna-ai[bot]@users.noreply.github.com" in commit
    monkeypatch.delenv("CODNA_GIT_USER_NAME")
    monkeypatch.delenv("CODNA_GIT_USER_EMAIL")
    assert pg._git_identity_args() == ["-c", "user.email=codna@codna.ai", "-c", "user.name=codna"]
