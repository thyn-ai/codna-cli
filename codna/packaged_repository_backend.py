from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .packaged_git import MaterializedRepository, materialize_repository
from .packaged_repository_advanced import PackagedAgentRunner, PackagedRepositoryAdvanced

SCHEMA_VERSION = 1
BACKEND_MODE = "packaged_local_triage"
MAX_SAMPLE_BYTES = 64 * 1024
MAX_SNIPPET_CHARS = 900
SNIPPET_CONTEXT_BEFORE = 4
SNIPPET_CONTEXT_AFTER = 6
MAX_EVIDENCE_ITEMS = 10
TOKEN_BYTES = 4

IGNORED_DIR_NAMES = {
    ".codna",
    ".codna-memory",
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".turbo",
    ".venv",
    ".vscode",
    "__pycache__",
    "bower_components",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "out",
    "target",
    "venv",
}

BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".bin",
    ".bmp",
    ".class",
    ".dll",
    ".dmg",
    ".doc",
    ".docx",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mp3",
    ".mp4",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".rar",
    ".so",
    ".tar",
    ".wasm",
    ".webp",
    ".whl",
    ".zip",
}

LANGUAGE_BY_SUFFIX = {
    ".adb": "ada",
    ".ads": "ada",
    ".astro": "astro",
    ".awk": "awk",
    ".bash": "shell",
    ".bat": "batch",
    ".blade.php": "blade",
    ".c": "c",
    ".cc": "cpp",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cmake": "cmake",
    ".coffee": "coffeescript",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".cts": "typescript",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".cxx": "cpp",
    ".dart": "dart",
    ".dockerfile": "dockerfile",
    ".eex": "elixir",
    ".elm": "elm",
    ".erl": "erlang",
    ".ex": "elixir",
    ".exs": "elixir",
    ".f90": "fortran",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".go": "go",
    ".graphql": "graphql",
    ".groovy": "groovy",
    ".h": "c",
    ".handlebars": "handlebars",
    ".hcl": "hcl",
    ".heex": "elixir",
    ".hpp": "cpp",
    ".hs": "haskell",
    ".html": "html",
    ".java": "java",
    ".jl": "julia",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".less": "less",
    ".liquid": "liquid",
    ".lua": "lua",
    ".m": "objective-c",
    ".md": "markdown",
    ".mdx": "mdx",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".mm": "objective-cpp",
    ".mts": "typescript",
    ".nim": "nim",
    ".nix": "nix",
    ".php": "php",
    ".pl": "perl",
    ".proto": "protobuf",
    ".ps1": "powershell",
    ".py": "python",
    ".pyi": "python",
    ".r": "r",
    ".rb": "ruby",
    ".re": "reasonml",
    ".res": "rescript",
    ".rs": "rust",
    ".scala": "scala",
    ".scss": "scss",
    ".sh": "shell",
    ".svelte": "svelte",
    ".swift": "swift",
    ".tf": "terraform",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".vue": "vue",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zig": "zig",
}

LANGUAGE_BY_FILENAME = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "Rakefile": "ruby",
    "Tiltfile": "starlark",
    "WORKSPACE": "starlark",
    "BUILD": "starlark",
}

CODE_LANGUAGES = {
    "ada",
    "astro",
    "awk",
    "batch",
    "c",
    "clojure",
    "coffeescript",
    "cpp",
    "csharp",
    "cuda",
    "dart",
    "elixir",
    "elm",
    "erlang",
    "fortran",
    "fsharp",
    "go",
    "groovy",
    "haskell",
    "java",
    "javascript",
    "julia",
    "kotlin",
    "lua",
    "nim",
    "objective-c",
    "objective-cpp",
    "ocaml",
    "perl",
    "php",
    "powershell",
    "python",
    "r",
    "reasonml",
    "rescript",
    "ruby",
    "rust",
    "scala",
    "shell",
    "starlark",
    "swift",
    "typescript",
    "zig",
}

TERM_RE = re.compile(r"[A-Za-z0-9_./-]{3,}")
SYMBOL_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|const|let|var|export\s+function|"
    r"func|fn|pub\s+fn|public\s+class|private\s+class)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


class PackagedRepositoryBackendError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class SnippetSelection:
    snippet: str
    symbol: str | None


@dataclass(frozen=True)
class FileEntry:
    path: str
    size_bytes: int
    language: str
    token_estimate: int
    sample: str
    sample_sha256: str


class PackagedRepositoryBackend:
    """Dependency-light local repository intelligence for clean Codna installs.

    This is not a mock. It gives users a real local snapshot, triage, sidecar
    fix, deterministic patch simulation, and explicit local apply path when the
    full Algenta app backend is not packaged.
    """

    def __init__(self, runtime_root: Path, *, agent_runner: PackagedAgentRunner | None = None) -> None:
        self._root = runtime_root.expanduser().resolve()
        self._snapshot_dir = self._root / "repository-intelligence" / "packaged" / "snapshots"
        self._bundle_dir = self._root / "repository-intelligence" / "packaged" / "bundles"
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._repositories: dict[str, MaterializedRepository] = {}
        self._advanced = PackagedRepositoryAdvanced(
            runtime_root=self._root,
            agent_runner=agent_runner,
        )

    def validate_connector_type(self, connector_type: str) -> None:
        if connector_type not in {"local_repo", "github_repo", "repo_archive"}:
            raise PackagedRepositoryBackendError(
                "invalid_connector_type",
                f"Connector type '{connector_type}' is not supported by local repository intelligence.",
                {"connector_type": connector_type},
            )

    def get_repository_intelligence_capabilities(self) -> dict[str, Any]:
        languages = sorted(set(LANGUAGE_BY_SUFFIX.values()) | set(LANGUAGE_BY_FILENAME.values()))
        return {
            "backend_mode": BACKEND_MODE,
            "supported_languages": languages,
            "support_progress": {
                "supported_real_language_count": len(languages),
                "ranked_target_language_count": len(languages),
                "progress_fraction": 1.0,
                "progress_label": f"{len(languages)} / {len(languages)} packaged",
            },
            "advanced_operations": "full_algenta_backend_required",
        }

    def create_repository_snapshot(
        self,
        *,
        connector_record: Any,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        materialized = self._repo_root(connector_record, request)
        repo_root = materialized.path
        focus_paths = self._validated_path_list(request.get("focus_paths") or [], repo_root)
        entries = self._scan_repo(repo_root)
        language_counts: dict[str, int] = {}
        raw_tokens = 0
        inventory_rows: list[dict[str, Any]] = []
        content_hasher = hashlib.sha256()
        for entry in entries:
            language_counts[entry.language] = language_counts.get(entry.language, 0) + 1
            raw_tokens += entry.token_estimate
            content_hasher.update(entry.path.encode("utf-8"))
            content_hasher.update(b"\0")
            content_hasher.update(str(entry.size_bytes).encode("ascii"))
            content_hasher.update(b"\0")
            content_hasher.update(entry.sample_sha256.encode("ascii"))
            content_hasher.update(b"\0")
            inventory_rows.append(
                {
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "language": entry.language,
                    "token_estimate": entry.token_estimate,
                    "sample": entry.sample,
                    "sample_sha256": entry.sample_sha256,
                }
            )
        content_hash = content_hasher.hexdigest()
        snapshot_id = "snap_" + _sha256_text(
            "|".join([str(connector_record.id), content_hash, ",".join(focus_paths)])
        )[:16]
        created_at = _utc_now()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": connector_record.id,
            "snapshot_id": snapshot_id,
            "connector_type": connector_record.connector_type,
            "_repo_root_path": str(repo_root),
            "repository_url": materialized.repository_url,
            "repository_slug": materialized.repository_slug,
            "resolved_revision": str(
                connector_record.config.get("resolved_revision") or _git_revision(repo_root) or "local"
            ),
            "content_hash": content_hash,
            "status": "ready",
            "created_at": created_at,
            "file_count": len(entries),
            "snapshot_file_count": len(entries),
            "language_counts": dict(sorted(language_counts.items())),
            "raw_repo_token_estimate": raw_tokens,
            "focus_paths": focus_paths,
            "ref": request.get("ref") if isinstance(request.get("ref"), str) else None,
            "files": inventory_rows,
        }
        snapshot_path = self._snapshot_dir / f"{snapshot_id}.json"
        artifact = self._write_json_artifact(snapshot_path, payload, "repository_snapshot")
        public_payload = dict(payload)
        public_payload.pop("files", None)
        public_payload.pop("_repo_root_path", None)
        public_payload["repository_snapshot_artifact"] = artifact
        public_payload["repository_graph_artifact"] = _empty_artifact("repository_graph", created_at)
        public_payload["symbol_graph_artifact"] = _empty_artifact("symbol_graph", created_at)
        public_payload["dependency_graph_artifact"] = _empty_artifact("dependency_graph", created_at)
        self._snapshots[snapshot_id] = payload
        return public_payload

    def get_repository_snapshot(self, *, repository_id: str, snapshot_id: str) -> dict[str, Any]:
        payload = self._load_snapshot(snapshot_id)
        if payload.get("repository_id") != repository_id:
            raise PackagedRepositoryBackendError(
                "unknown_repository_snapshot",
                f"Snapshot '{snapshot_id}' is not registered for repository '{repository_id}'.",
                {"repository_id": repository_id, "snapshot_id": snapshot_id},
            )
        created_at = str(payload.get("created_at") or _utc_now())
        output = dict(payload)
        output.pop("files", None)
        output.pop("_repo_root_path", None)
        output["repository_snapshot_artifact"] = _artifact_for_path(
            self._snapshot_dir / f"{snapshot_id}.json",
            "repository_snapshot",
            created_at,
        )
        output["repository_graph_artifact"] = _empty_artifact("repository_graph", created_at)
        output["symbol_graph_artifact"] = _empty_artifact("symbol_graph", created_at)
        output["dependency_graph_artifact"] = _empty_artifact("dependency_graph", created_at)
        return output

    def triage_repository(self, *, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        snapshot_id = _required_string(request, "snapshot_id")
        snapshot = self._load_snapshot(snapshot_id)
        if snapshot.get("repository_id") != repository_id:
            raise PackagedRepositoryBackendError(
                "unknown_repository_snapshot",
                f"Snapshot '{snapshot_id}' is not registered for repository '{repository_id}'.",
                {"repository_id": repository_id, "snapshot_id": snapshot_id},
            )
        signals = request.get("signals") if isinstance(request.get("signals"), dict) else {}
        query = _query_text(signals)
        changed_files = _string_list(signals.get("changed_files"))
        ranked = self._rank_files(
            snapshot,
            query,
            changed_files,
            test_driven=_is_test_driven_fix(signals),
        )
        evidence_items = [
            _evidence_item(rank=index + 1, entry=entry, score=score, query=query)
            for index, (entry, score) in enumerate(ranked[:MAX_EVIDENCE_ITEMS])
        ]
        suspect_files = [item["file_path"] for item in evidence_items if item.get("file_path")]
        suspect_symbols = _suspect_symbols(evidence_items)
        evidence_tokens = max(1, sum(int(item["token_count"]) for item in evidence_items))
        raw_tokens = int(snapshot.get("raw_repo_token_estimate") or 0)
        reduction_ratio = float(raw_tokens / evidence_tokens) if evidence_tokens else 0.0
        created_at = _utc_now()
        bundle_ref = "packaged-local://" + _sha256_text(
            json.dumps(
                {
                    "snapshot_id": snapshot_id,
                    "query": query,
                    "suspect_files": suspect_files,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )[:24]
        bundle_payload = {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "snapshot_id": snapshot_id,
            "workspace_evidence_bundle_ref": bundle_ref,
            "created_at": created_at,
            "query": query,
            "raw_repo_token_estimate": raw_tokens,
            "evidence_bundle_token_count": evidence_tokens,
            "reduction_ratio": reduction_ratio,
            "suspect_files": suspect_files,
            "suspect_symbols": suspect_symbols,
            "evidence_items": evidence_items,
        }
        bundle_path = self._bundle_dir / f"{bundle_ref.rsplit('/', 1)[-1]}.json"
        bundle_artifact = self._write_json_artifact(bundle_path, bundle_payload, "workspace_evidence_bundle")
        return {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "snapshot_id": snapshot_id,
            "workspace_evidence_bundle_ref": bundle_ref,
            "created_at": created_at,
            "raw_repo_token_estimate": raw_tokens,
            "evidence_bundle_token_count": evidence_tokens,
            "reduction_ratio": reduction_ratio,
            "evidence_items": evidence_items,
            "workspace_evidence_bundle_artifact": bundle_artifact,
            "suspect_files": suspect_files,
            "suspect_symbols": suspect_symbols,
        }

    def query_repository_graph(self, *, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        snapshot_id = _required_string(request, "snapshot_id")
        snapshot = self._load_snapshot(snapshot_id)
        seed_paths = _string_list(request.get("seed_file_paths"))
        if not seed_paths:
            seed_paths = _string_list(request.get("changed_files"))
        known = {str(item.get("path")) for item in snapshot.get("files", []) if isinstance(item, dict)}
        impacted = [path for path in seed_paths if path in known]
        created_at = _utc_now()
        return {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "seed_file_paths": impacted,
            "seed_symbols": [],
            "direct_dependencies": [],
            "direct_dependents": [],
            "impacted_files": impacted,
            "impacted_symbols": [],
            "graph_nodes": [
                {
                    "file_path": path,
                    "depth": 0,
                    "is_seed": True,
                    "inbound_count": 0,
                    "outbound_count": 0,
                    "risk_score": 0.0,
                    "contained_symbols": [],
                    "parent_child_symbols": [],
                }
                for path in impacted
            ],
            "graph_edges": [],
            "top_change_risk_files": [
                {
                    "file_path": path,
                    "depth": 0,
                    "relationship": "seed",
                    "risk_score": 0.0,
                    "top_symbol": None,
                }
                for path in impacted
            ],
        }

    def create_repository_decision_plan(self, *, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        snapshot_id = _required_string(request, "snapshot_id")
        bundle_ref = _required_string(request, "workspace_evidence_bundle_ref")
        return self._advanced.create_repository_decision_plan(
            repository_id=repository_id,
            request=request,
            snapshot=self._load_snapshot(snapshot_id),
            evidence_bundle=self._load_bundle(bundle_ref),
        )

    def simulate_repository(self, *, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._advanced.simulate_repository(repository_id=repository_id, request=request)

    def apply_repository(self, *, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        enriched_request = dict(request)
        materialized = self._repositories.get(repository_id)
        if materialized is not None and materialized.repository_url:
            enriched_request.setdefault("repository_url", materialized.repository_url)
            enriched_request.setdefault("repository_slug", materialized.repository_slug)
            enriched_request.setdefault("access_token", materialized.access_token)
        return self._advanced.apply_repository(repository_id=repository_id, request=enriched_request)

    def _repo_root(self, connector_record: Any, request: Mapping[str, Any]) -> MaterializedRepository:
        materialized = materialize_repository(
            runtime_root=self._root,
            connector_id=str(connector_record.id),
            connector_type=str(connector_record.connector_type),
            config=connector_record.config,
            request=request,
        )
        self._repositories[str(connector_record.id)] = materialized
        return materialized

    def _scan_repo(self, repo_root: Path) -> list[FileEntry]:
        entries: list[FileEntry] = []
        for path in _iter_repo_files(repo_root):
            entry = _file_entry(repo_root, path)
            if entry is not None:
                entries.append(entry)
        return entries

    def _rank_files(
        self,
        snapshot: Mapping[str, Any],
        query: str,
        changed_files: list[str],
        *,
        test_driven: bool = False,
    ) -> list[tuple[FileEntry, float]]:
        changed = set(changed_files) | set(_string_list(snapshot.get("focus_paths")))
        terms = _query_terms(query)
        ranked: list[tuple[FileEntry, float]] = []
        for item in snapshot.get("files", []):
            if not isinstance(item, dict):
                continue
            entry = FileEntry(
                path=str(item.get("path") or ""),
                size_bytes=int(item.get("size_bytes") or 0),
                language=str(item.get("language") or "unknown"),
                token_estimate=int(item.get("token_estimate") or 0),
                sample=str(item.get("sample") or ""),
                sample_sha256=str(item.get("sample_sha256") or ""),
            )
            ranked.append((entry, _score_entry(entry, terms, changed, test_driven=test_driven)))
        return sorted(ranked, key=lambda row: (-row[1], row[0].path))

    def _load_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        cached = self._snapshots.get(snapshot_id)
        if cached is not None:
            return dict(cached)
        path = self._snapshot_dir / f"{snapshot_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PackagedRepositoryBackendError(
                "unknown_repository_snapshot",
                f"Snapshot '{snapshot_id}' is not registered in the packaged local backend.",
                {"snapshot_id": snapshot_id, "storage_path": str(path)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise PackagedRepositoryBackendError(
                "corrupt_repository_snapshot",
                f"Snapshot '{snapshot_id}' is not valid JSON.",
                {"snapshot_id": snapshot_id, "storage_path": str(path)},
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise PackagedRepositoryBackendError(
                "corrupt_repository_snapshot",
                f"Snapshot '{snapshot_id}' has an unsupported schema.",
                {"snapshot_id": snapshot_id, "storage_path": str(path)},
            )
        self._snapshots[snapshot_id] = dict(payload)
        return payload

    def _load_bundle(self, bundle_ref: str) -> dict[str, Any]:
        bundle_id = bundle_ref.rsplit("/", 1)[-1]
        path = self._bundle_dir / f"{bundle_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PackagedRepositoryBackendError(
                "unknown_workspace_evidence_bundle",
                "Workspace evidence bundle is not registered in the packaged local backend.",
                {"workspace_evidence_bundle_ref": bundle_ref, "storage_path": str(path)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise PackagedRepositoryBackendError(
                "corrupt_workspace_evidence_bundle",
                "Workspace evidence bundle is not valid JSON.",
                {"workspace_evidence_bundle_ref": bundle_ref, "storage_path": str(path)},
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("workspace_evidence_bundle_ref") != bundle_ref
        ):
            raise PackagedRepositoryBackendError(
                "corrupt_workspace_evidence_bundle",
                "Workspace evidence bundle has an unsupported schema.",
                {"workspace_evidence_bundle_ref": bundle_ref, "storage_path": str(path)},
            )
        return payload

    def _write_json_artifact(self, path: Path, payload: Mapping[str, Any], artifact_kind: str) -> dict[str, Any]:
        _atomic_write_json(path, payload)
        return _artifact_for_path(path, artifact_kind, str(payload.get("created_at") or _utc_now()))

    @staticmethod
    def _validated_path_list(value: Any, repo_root: Path) -> list[str]:
        paths = _string_list(value)
        output: list[str] = []
        seen: set[str] = set()
        for item in paths:
            normalized = _safe_relative_path(repo_root, item)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output


def _iter_repo_files(repo_root: Path) -> list[Path]:
    output: list[Path] = []
    for current_root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIR_NAMES)
        current = Path(current_root)
        for filename in sorted(filenames):
            path = current / filename
            if path.is_file() and not path.is_symlink():
                output.append(path)
    return output


def _file_entry(repo_root: Path, path: Path) -> FileEntry | None:
    suffix = _normalized_suffix(path)
    if suffix in BINARY_SUFFIXES:
        return None
    try:
        stat = path.stat()
        sample_bytes = path.read_bytes()[:MAX_SAMPLE_BYTES]
    except OSError:
        return None
    if b"\0" in sample_bytes:
        return None
    relative = path.relative_to(repo_root).as_posix()
    language = _language_for_path(path)
    sample = sample_bytes.decode("utf-8", errors="replace")
    sample_sha = hashlib.sha256(sample_bytes).hexdigest()
    token_estimate = max(1, stat.st_size // TOKEN_BYTES)
    return FileEntry(
        path=relative,
        size_bytes=stat.st_size,
        language=language,
        token_estimate=token_estimate,
        sample=sample,
        sample_sha256=sample_sha,
    )


def _language_for_path(path: Path) -> str:
    if path.name in LANGUAGE_BY_FILENAME:
        return LANGUAGE_BY_FILENAME[path.name]
    suffix = _normalized_suffix(path)
    return LANGUAGE_BY_SUFFIX.get(suffix, suffix.lstrip(".") or "text")


def _normalized_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".blade.php"):
        return ".blade.php"
    return path.suffix.lower()


def _score_entry(entry: FileEntry, terms: set[str], changed_files: set[str], *, test_driven: bool = False) -> float:
    if entry.path in changed_files:
        return 10_000.0
    path_text = entry.path.lower().replace("_", "-")
    sample_text = entry.sample.lower()
    score = 0.0
    for term in terms:
        normalized = term.lower().replace("_", "-")
        if normalized in path_text:
            score += 120.0
        if normalized in sample_text:
            score += min(80.0, sample_text.count(normalized) * 8.0)
    if entry.language in CODE_LANGUAGES:
        score += 8.0
    elif entry.language in {"markdown", "mdx"}:
        score -= 12.0
    if any(part in {"src", "lib", "app", "packages", "cmd", "internal"} for part in entry.path.split("/")):
        score += 12.0
    if _is_test_path(entry.path):
        score -= 4.0
        if test_driven:
            score -= 180.0
    if entry.path.endswith((".json", ".lock", ".snap")):
        score -= 10.0
    return score


def _is_test_driven_fix(signals: Mapping[str, Any]) -> bool:
    if _string_list(signals.get("failing_tests")):
        return True
    issue = signals.get("issue_text")
    return isinstance(issue, str) and bool(
        re.search(r"\b(failing test|tests? are failing|unittest|pytest|junit|assertionerror)\b", issue, re.I)
    )


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    name = parts[-1] if parts else normalized
    return (
        any(part in {"test", "tests", "__tests__", "spec", "specs"} for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", "_spec.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
    )


def _evidence_item(*, rank: int, entry: FileEntry, score: float, query: str) -> dict[str, Any]:
    selection = _snippet_selection(entry.sample, _query_terms(query))
    snippet = selection.snippet
    return {
        "evidence_id": "ev_" + _sha256_text(f"{rank}|{entry.path}|{entry.sample_sha256}")[:16],
        "rank": rank,
        "source_type": "file",
        "source_ref": entry.path,
        "summary": f"{entry.path} ({entry.language}, {entry.token_estimate} estimated tokens)",
        "snippet": snippet,
        "token_count": min(max(1, entry.token_estimate), max(1, len(snippet.encode("utf-8")) // TOKEN_BYTES)),
        "score": float(score),
        "file_path": entry.path,
        "symbol_name": selection.symbol,
    }


def _snippet_selection(sample: str, terms: set[str]) -> SnippetSelection:
    lines = [line.rstrip() for line in sample.splitlines()]
    if not lines:
        return SnippetSelection(snippet="", symbol=None)
    selected_index = _best_snippet_line_index(lines, terms)
    snippet = _numbered_context_window(lines, selected_index)
    if len(snippet) > MAX_SNIPPET_CHARS:
        snippet = snippet[: MAX_SNIPPET_CHARS - 3] + "..."
    return SnippetSelection(snippet=snippet, symbol=_nearest_symbol(lines, selected_index))


def _best_snippet_line_index(lines: list[str], terms: set[str]) -> int:
    best_index = -1
    best_score = 0
    for index, line in enumerate(lines):
        score = _snippet_line_score(line, terms)
        if score > best_score:
            best_score = score
            best_index = index
    if best_index >= 0:
        return best_index
    for index, line in enumerate(lines):
        if line.strip():
            return index
    return 0


def _snippet_line_score(line: str, terms: set[str]) -> int:
    stripped = line.strip()
    lower = stripped.lower()
    score = 0
    for term in terms:
        normalized = term.lower()
        count = lower.count(normalized)
        if count:
            score += count * (3 + min(len(normalized) // 4, 4))
    if score and SYMBOL_RE.match(line):
        score += 10
    if score and stripped.startswith(("return ", "raise ", "assert ")):
        score += 3
    if score and stripped.startswith(("import ", "from ")):
        score = max(1, score - 5)
    return score


def _numbered_context_window(lines: list[str], selected_index: int) -> str:
    start = max(0, selected_index - SNIPPET_CONTEXT_BEFORE)
    end = min(len(lines), selected_index + SNIPPET_CONTEXT_AFTER + 1)
    return "\n".join(f"{index + 1:6}\t{lines[index]}" for index in range(start, end))


def _nearest_symbol(lines: list[str], selected_index: int) -> str | None:
    for index in range(selected_index, -1, -1):
        match = SYMBOL_RE.match(lines[index])
        if match:
            return match.group(1)
    for index in range(selected_index + 1, len(lines)):
        match = SYMBOL_RE.match(lines[index])
        if match:
            return match.group(1)
    return None


def _suspect_symbols(evidence_items: list[dict[str, Any]]) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for item in evidence_items:
        symbol = item.get("symbol_name")
        if isinstance(symbol, str) and symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _query_text(signals: Mapping[str, Any]) -> str:
    parts: list[str] = []
    issue = signals.get("issue_text")
    if isinstance(issue, str):
        parts.append(issue)
    for key in ("failing_tests", "changed_files"):
        for item in _string_list(signals.get(key)):
            parts.append(item)
    return " ".join(parts).strip()


def _query_terms(query: str) -> set[str]:
    return {
        term.strip("./-").lower()
        for term in TERM_RE.findall(query)
        if len(term.strip("./-")) >= 3
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PackagedRepositoryBackendError(
            "invalid_repository_request",
            f"Repository request requires a non-empty {key}.",
            {"field": key},
        )
    return value


def _safe_relative_path(repo_root: Path, raw_path: str) -> str | None:
    candidate = raw_path.strip().strip("`'\"")
    candidate = candidate.lstrip("<([{").rstrip(">)]},:;")
    candidate = candidate.replace("\\", "/")
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(repo_root)
        except ValueError:
            return None
    else:
        relative = Path(PurePosixPath(candidate))
        if relative.is_absolute() or ".." in relative.parts:
            return None
    return PurePosixPath(relative.as_posix()).as_posix()


def _artifact_for_path(path: Path, artifact_kind: str, created_at: str) -> dict[str, Any]:
    return {
        "artifact_id": "artifact_" + _sha256_text(f"{artifact_kind}|{path}")[:16],
        "artifact_kind": artifact_kind,
        "content_hash": _sha256_file(path) if path.is_file() else _sha256_text(artifact_kind),
        "storage_path": str(path),
        "schema_revision": f"packaged-local-v{SCHEMA_VERSION}",
        "created_at": created_at,
    }


def _empty_artifact(artifact_kind: str, created_at: str) -> dict[str, Any]:
    return {
        "artifact_id": f"artifact_empty_{artifact_kind}",
        "artifact_kind": artifact_kind,
        "content_hash": _sha256_text(f"empty:{artifact_kind}"),
        "storage_path": "",
        "schema_revision": f"packaged-local-v{SCHEMA_VERSION}",
        "created_at": created_at,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = out.stdout.strip()
    return revision if out.returncode == 0 and revision else None
