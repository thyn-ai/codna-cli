"""Local content-hash manifest helpers for Codna memory indexing."""
from __future__ import annotations


def manifest_hash(entry) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        value = entry.get("hash")
        return value if isinstance(value, str) else None
    return None


def manifest_entry(*, digest: str, repo_id: str, service: str, language: str, scope_key: str) -> dict:
    return {
        "hash": digest,
        "repo_id": repo_id,
        "service": service,
        "language": language,
        "scope_key": scope_key,
    }


def manifest_scope(entry) -> str | None:
    if isinstance(entry, dict):
        value = entry.get("scope_key")
        return value if isinstance(value, str) else None
    return None


def legacy_manifest_id_belongs_to_repo(unit_id: str, repo_id: str) -> bool:
    return unit_id.startswith(f"{repo_id}:")
