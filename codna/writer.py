"""GitHubWriter — the PR-writer privilege domain.

The writer is the ONLY component that holds a write-capable token, and it NEVER executes
repository code. It receives the worker's patch + signed attestation, independently
re-verifies the handoff (signature, required fields, single-use nonce, patch-digest match),
revalidates that the base commit has not moved since the proof (TOCTOU defense, re-checked
at the last moment before opening), derives a collision/injection-safe branch name, and opens
a DRAFT PR via a narrowly-scoped token.

Side effects flow through two injected seams — `create_pr(...)` and `head_resolver(...)` — so
the verification logic is fully testable offline. The real seams (gh CLI / GitHub API) are
provided by the `*_factory` helpers and are the only place that touches the network/token.
"""
from __future__ import annotations

import re
from typing import Callable

from .evidence import Attestation, NonceStore, Signer, verify_attestation
from .findings import IngestResult, NormalizedFinding, digest_of
from .secure import Patch

_BRANCH_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]")


class WriterError(Exception):
    pass


class GitHubWriter:
    def __init__(
        self,
        *,
        trusted_signers: dict[str, Signer],
        create_pr: Callable[..., str],
        head_resolver: Callable[[str], str],
        base_branch: str = "main",
        nonce_store: NonceStore | None = None,
        draft: bool = True,
        title_for: Callable[[NormalizedFinding], str] | None = None,
        body_for: Callable[[NormalizedFinding, Attestation], str] | None = None,
    ):
        self._trusted = dict(trusted_signers)
        self._create_pr = create_pr
        self._head_resolver = head_resolver
        self._base_branch = base_branch
        self._nonce_store = nonce_store or NonceStore()
        self._draft = draft
        self._title_for = title_for or (lambda f: f"codna: fix {f.rule_id}")
        self._body_for = body_for or self._default_body
        self.opened: list[str] = []

    def base_unchanged(self, ingest: IngestResult) -> bool:
        return self._head_resolver(self._base_branch) == ingest.commit

    def _branch_for(self, finding: NormalizedFinding) -> str:
        # Strict charset — no spaces, slashes (beyond our prefix), or traversal can reach the
        # ref, so a hostile canonical id can't inject a branch/ref (WRITER-REF).
        slug = _BRANCH_UNSAFE.sub("-", finding.canonical_id.replace("sha256:", ""))[:60].strip("-.")
        return f"codna/secure/{slug or 'finding'}"

    def open_draft_pr(self, finding: NormalizedFinding, patch: Patch, attestation: Attestation) -> str:
        ok, reason = verify_attestation(
            attestation, trusted_signers=self._trusted, nonce_store=self._nonce_store
        )
        if not ok:
            raise WriterError(f"attestation rejected: {reason}")

        payload = attestation.payload
        if payload.get("patch_digest") != digest_of(patch.diff):
            raise WriterError("patch digest does not match the attested digest")
        # Re-resolve the base at the LAST moment — a proof against a moved base is invalid.
        if payload.get("original_commit") != self._head_resolver(self._base_branch):
            raise WriterError("base commit changed since the proof (stale tree)")

        url = self._create_pr(
            branch=self._branch_for(finding),
            base=self._base_branch,
            title=self._title_for(finding),
            body=self._body_for(finding, attestation),
            patch_diff=patch.diff,
            draft=self._draft,
        )
        self.opened.append(url)
        return url

    @staticmethod
    def _default_body(finding: NormalizedFinding, attestation: Attestation) -> str:
        p = attestation.payload
        return (
            "Automated, evidence-backed security fix by **codna**.\n\n"
            f"- finding: `{finding.rule_id}` ({finding.finding_kind.value})\n"
            f"- proof: `{p.get('proof_results')}`\n"
            f"- original commit: `{p.get('original_commit')}`\n"
            f"- patch digest: `{p.get('patch_digest')}`\n"
            f"- resulting tree: `{p.get('resulting_tree')}`\n"
            f"- mojo verdict: `{p.get('mojo_verdict')}`\n\n"
            "_Draft — the proof closed the obligation without unacceptable regression risk; "
            "review before merging._"
        )


def git_push_create_pr_factory(repo_dir: str, slug: str, token: str) -> Callable[..., str]:
    """Real create_pr: push a branch built from the patch (via GitBranchBuilder) using a
    token-embedded URL, then open the draft PR over the API. The git push is plumbing only —
    the project's code is never executed here."""
    import httpx

    from .gitpush import GitBranchBuilder

    builder = GitBranchBuilder(repo_dir)
    push_url = f"https://x-access-token:{token}@github.com/{slug}.git"
    headers = {"authorization": f"Bearer {token}", "accept": "application/vnd.github+json"}

    def create_pr(*, branch: str, base: str, title: str, body: str, patch_diff: str, draft: bool) -> str:
        builder.build_and_push(branch=branch, base=base, patch_diff=patch_diff,
                               commit_message=title, push_url=push_url)
        r = httpx.post(f"https://api.github.com/repos/{slug}/pulls", headers=headers, timeout=60,
                       json={"title": title, "head": branch, "base": base, "body": body, "draft": draft})
        if r.status_code >= 400:
            raise WriterError(f"open PR -> {r.status_code} {r.text[:200]}")
        return r.json().get("html_url", "")

    return create_pr


def cloned_push_pr_factory(slug: str, token: str, base_branch: str) -> tuple[Callable[..., str], Callable[[], None]]:
    """Clone `base_branch` of `slug` into a throwaway checkout and return ``(create_pr, cleanup)``:
    a git-backed PR factory that applies the verified diff onto REAL files and pushes a real branch
    (not the empty-PR API stub), plus a ``cleanup()`` to remove the checkout. Git plumbing only —
    the project's code is never executed here."""
    import os
    import shutil
    import subprocess
    import tempfile

    workdir = tempfile.mkdtemp(prefix="codna-secure-pr-")
    repo_dir = os.path.join(workdir, "repo")

    def cleanup() -> None:
        shutil.rmtree(workdir, ignore_errors=True)

    clone_url = f"https://x-access-token:{token}@github.com/{slug}.git"
    cl = subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", "--branch", base_branch, clone_url, repo_dir],
        capture_output=True, text=True,
    )
    if cl.returncode != 0:
        cleanup()
        raise WriterError(f"could not clone {slug}@{base_branch} to build the fix branch: {cl.stderr.strip()[:200]}")
    return git_push_create_pr_factory(repo_dir, slug, token), cleanup


def gh_head_resolver_factory(repo: str, api_key: str | None) -> Callable[[str], str]:
    """Resolve a branch's head sha via the GitHub API (read-only). Lazy httpx import."""
    import httpx

    def resolve(base_branch: str) -> str:
        headers = {"accept": "application/vnd.github+json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        r = httpx.get(f"https://api.github.com/repos/{repo}/commits/{base_branch}",
                      headers=headers, timeout=60)
        if r.status_code >= 400:
            raise WriterError(f"head resolve {repo}@{base_branch} -> {r.status_code}")
        return r.json().get("sha", "")

    return resolve
