"""CLI adapter for the privilege-separated security evidence writer."""
from __future__ import annotations

import json
import re
from pathlib import Path


class SecureOpenPrCliError(ValueError):
    pass


def _safe_evidence_bundle_name(canonical_id: str) -> str:
    value = canonical_id.replace("sha256:", "")
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", value)[:64].strip(".-")
    return safe or "finding"


def write_secure_evidence(evidence_dir: str, bundles) -> int:
    from .evidence import write_evidence

    root = Path(evidence_dir).expanduser()
    if root.exists() and any(root.iterdir()):
        raise SecureOpenPrCliError(f"evidence dir already exists and is not empty: {root}. Use a fresh directory.")
    root.mkdir(parents=True, exist_ok=True)

    entries = []
    single_bundle = len(bundles) == 1
    for index, bundle in enumerate(bundles, start=1):
        rel_path = "." if single_bundle else f"{index:03d}-{_safe_evidence_bundle_name(bundle.finding.canonical_id)}"
        target = root if single_bundle else root / rel_path
        write_evidence(
            str(target),
            attestation=bundle.attestation,
            patch_diff=bundle.patch.diff,
            finding=bundle.finding,
        )
        entries.append({
            "path": rel_path,
            "canonical_id": bundle.finding.canonical_id,
            "rule_id": bundle.finding.rule_id,
        })

    index_path = root / "evidence-index.json"
    index_path.write_text(json.dumps({"schema_version": 1, "bundles": entries}, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(f"  evidence: wrote {len(entries)} bundle(s) to {root}")
    return len(entries)


def evidence_bundle_dirs(evidence_dir: str) -> list[Path]:
    root = Path(evidence_dir).expanduser()
    if root.is_dir() and not any(root.iterdir()):
        return []
    if (root / "attestation.json").is_file():
        return [root]
    index_path = root / "evidence-index.json"
    if not index_path.is_file():
        return [root]
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SecureOpenPrCliError(f"could not parse evidence index {index_path}: {exc}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("bundles"), list):
        raise SecureOpenPrCliError(f"evidence index has unsupported schema: {index_path}")
    paths: list[Path] = []
    for entry in payload["bundles"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise SecureOpenPrCliError(f"evidence index has an invalid bundle entry: {index_path}")
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise SecureOpenPrCliError(f"evidence index contains an unsafe bundle path: {entry['path']}")
        paths.append(root if entry["path"] == "." else root / rel)
    return paths


def preflight_evidence(evidence: list[tuple[object, str, object]], key_bytes: bytes):
    """Verify evidence signatures and patch digests before any GitHub clone/push setup."""
    from .evidence import HmacSigner, NonceStore, verify_attestation
    from .findings import digest_of

    signers = {att.signer_id: HmacSigner(key_bytes, key_id=att.signer_id) for att, _, _ in evidence}
    nonce_store = NonceStore()
    for att, diff, _finding in evidence:
        ok, reason = verify_attestation(att, trusted_signers=signers, nonce_store=nonce_store)
        if not ok:
            raise SecureOpenPrCliError(f"evidence attestation rejected: {reason}")
        if att.payload.get("patch_digest") != digest_of(diff):
            raise SecureOpenPrCliError("patch digest does not match the attested digest")
    return signers


def cmd_secure_open_pr(args, *, die, clone_factory=None, head_resolver_factory=None) -> None:
    """Verify worker evidence and open draft PRs without executing repository code."""
    import os

    from .evidence import read_evidence
    from .findings import NormalizedFinding
    from .secure import Patch
    from .writer import GitHubWriter, WriterError, cloned_push_pr_factory, gh_head_resolver_factory

    evidence = []
    try:
        bundle_dirs = evidence_bundle_dirs(args.evidence)
    except SecureOpenPrCliError as exc:
        die(str(exc))
    for bundle_dir in bundle_dirs:
        try:
            att, diff, desc = read_evidence(str(bundle_dir))
        except Exception as exc:  # noqa: BLE001 - evidence is external input; never leak tracebacks.
            die(f"could not read evidence bundle {bundle_dir}: {exc}")
        if not desc:
            die(f"evidence bundle {bundle_dir} is missing finding.json (cannot derive the branch).")
        finding = NormalizedFinding.minimal(canonical_id=desc["canonical_id"], rule_id=desc["rule_id"],
                                            finding_kind=desc["finding_kind"])
        evidence.append((att, diff, finding))
    if not evidence:
        print("codna: no evidence bundles; no PR opened")
        return

    key_bytes = (os.environ.get("CODNA_ATTESTATION_KEY") or "").encode()
    if not key_bytes:
        die("set CODNA_ATTESTATION_KEY (shared with the worker) to verify the attestation.")
    gh = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not gh:
        die("secure-open-pr needs a write token: --github-token or $GITHUB_TOKEN.")
    if not args.repo_slug:
        die("secure-open-pr needs --repo-slug owner/repo.")

    try:
        signers = preflight_evidence(evidence, key_bytes)
    except SecureOpenPrCliError as exc:
        die(str(exc))

    base_branch = args.base_branch or "main"
    clone_factory = clone_factory or cloned_push_pr_factory
    head_resolver_factory = head_resolver_factory or gh_head_resolver_factory
    try:
        create_pr, cleanup = clone_factory(args.repo_slug, gh, base_branch)
    except WriterError as exc:
        die(str(exc))
    writer = GitHubWriter(trusted_signers=signers, create_pr=create_pr,
                          head_resolver=head_resolver_factory(args.repo_slug, gh),
                          base_branch=base_branch)
    urls = []
    try:
        for att, diff, finding in evidence:
            urls.append(writer.open_draft_pr(finding, Patch(att.payload["patch_digest"], diff=diff), att))
    except Exception as exc:  # noqa: BLE001
        die(f"writer refused to open the PR: {exc}")
    finally:
        cleanup()
    for url in urls:
        print(f"✓ opened draft PR: {url}")
