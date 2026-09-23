from __future__ import annotations

import difflib
import importlib
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class LocalRepositoryShimError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


_PATCH_OLD_FILE_RE = re.compile(r"^--- (?:a/)?(.+)$")
_PATCH_NEW_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_PATCH_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
_LOCAL_SDK_JS_LIKE_SUFFIXES = frozenset({".mjs", ".cjs", ".mts", ".cts"})
_LOCAL_SDK_DIRECT_SUFFIX_LANGUAGES = {
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".xml": "xml",
    ".properties": "properties",
}
_GENERATED_DIFF_SKIP_DIRS = frozenset({
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
})
_GENERATED_DIFF_SKIP_FILES = frozenset({
    ".coverage",
    ".DS_Store",
    "coverage.xml",
})
_GENERATED_DIFF_SKIP_SUFFIXES = (".pyc", ".pyo")


def install_precise_patch_changed_files() -> None:
    """Patch Algenta's local SDK diff parser to count edited lines, not hunk context."""
    try:
        from apps.api_server.services import repository_intelligence_core as core
        from apps.api_server.services import repository_intelligence_core_bindings as bindings
        from apps.api_server.services import repository_intelligence_patch_analysis as patch_analysis
    except Exception as exc:  # noqa: BLE001
        raise LocalRepositoryShimError(
            "local_repository_patch_parser_install_failed",
            "Codna could not install the local repository patch parser shim.",
            {"reason": f"{type(exc).__name__}: {exc}"},
        ) from exc
    patch_analysis._patch_changed_files = precise_patch_changed_files
    bindings._patch_changed_files = precise_patch_changed_files
    core._patch_changed_files = precise_patch_changed_files


def install_repository_module_suffix_support() -> None:
    """Keep local Algenta SDK module-file handling aligned with modern JS/TS repos."""
    try:
        text_registry = _load_repository_module(
            "apps.api_server.services.repository_intelligence_text_file_registry"
        )
        file_selection = _load_repository_module(
            "apps.api_server.services.repository_intelligence_file_selection"
        )
        classification = _load_repository_module(
            "apps.api_server.services.repository_intelligence_indexing_classification"
        )
        indexing = _load_repository_module(
            "apps.api_server.services.repository_intelligence_indexing"
        )
        bindings = _load_repository_module(
            "apps.api_server.services.repository_intelligence_core_bindings"
        )
        core = _load_repository_module("apps.api_server.services.repository_intelligence_core")
    except Exception as exc:  # noqa: BLE001
        raise LocalRepositoryShimError(
            "local_repository_module_suffix_support_install_failed",
            "Codna could not install local repository module-suffix support.",
            {"reason": f"{type(exc).__name__}: {exc}"},
        ) from exc

    suffixes = frozenset(_LOCAL_SDK_DIRECT_SUFFIX_LANGUAGES)
    _extend_text_file_suffixes(text_registry, suffixes)
    _extend_text_file_suffixes(file_selection, suffixes)
    _extend_direct_language_map(classification)
    _extend_js_like_suffixes(indexing)
    for module in (classification, indexing, file_selection, bindings, core):
        _patch_language_for_path(module)


def install_verified_agentic_diff_compatibility() -> None:
    """Patch local Algenta verified-agentic behavior for Codna's local runtime contract."""
    try:
        verified_agentic = _load_repository_module(
            "apps.api_server.services.repository_intelligence_verified_agentic"
        )
        planner_generation = _load_repository_module(
            "apps.api_server.services.repository_intelligence_planner_generation"
        )
    except Exception as exc:  # noqa: BLE001
        raise LocalRepositoryShimError(
            "local_repository_verified_agentic_diff_shim_install_failed",
            "Codna could not install the local verified-agentic diff compatibility shim.",
            {"reason": f"{type(exc).__name__}: {exc}"},
        ) from exc

    current = getattr(verified_agentic, "_unified_diff_from_tree_states", None)
    if current is None:
        raise LocalRepositoryShimError(
            "local_repository_verified_agentic_diff_shim_missing_target",
            "Local Algenta verified-agentic module is missing _unified_diff_from_tree_states.",
            {"module": verified_agentic.__name__},
        )
    if not getattr(current, "_codna_git_apply_compatible", False):

        def unified_diff_from_tree_states(before: dict[str, str], after: dict[str, str]) -> str:
            return git_apply_compatible_unified_diff_from_tree_states(before, after)

        unified_diff_from_tree_states.__name__ = "_unified_diff_from_tree_states"
        unified_diff_from_tree_states.__qualname__ = "_unified_diff_from_tree_states"
        unified_diff_from_tree_states.__module__ = verified_agentic.__name__
        unified_diff_from_tree_states._codna_git_apply_compatible = True  # type: ignore[attr-defined]
        verified_agentic._unified_diff_from_tree_states = unified_diff_from_tree_states
        if hasattr(planner_generation, "_unified_diff_from_tree_states"):
            planner_generation._unified_diff_from_tree_states = unified_diff_from_tree_states
    _install_verified_agentic_agent_model_env_support(verified_agentic)


def _install_verified_agentic_agent_model_env_support(verified_agentic: ModuleType) -> None:
    original = getattr(verified_agentic, "_run_agent_with_failover", None)
    if original is None:
        raise LocalRepositoryShimError(
            "local_repository_verified_agentic_model_env_shim_missing_target",
            "Local Algenta verified-agentic module is missing _run_agent_with_failover.",
            {"module": verified_agentic.__name__},
        )
    if getattr(original, "_codna_agent_model_env_support", False):
        return

    async def run_agent_with_failover(
        run_agent,
        *,
        working_dir,
        task_spec,
        injected_context,
        limits,
        root_path,
    ):
        provider = os.environ.get("ALGENTA_AGENT_PROVIDER") or None
        model = os.environ.get("ALGENTA_AGENT_MODEL") or None
        if provider or model:
            task_spec = dict(task_spec)
            engine = task_spec.get("engine")
            engine = dict(engine) if isinstance(engine, dict) else {}
            task_spec["engine"] = engine
        return await original(
            run_agent,
            working_dir=working_dir,
            task_spec=task_spec,
            injected_context=injected_context,
            limits=limits,
            root_path=root_path,
        )

    run_agent_with_failover.__name__ = "_run_agent_with_failover"
    run_agent_with_failover.__qualname__ = "_run_agent_with_failover"
    run_agent_with_failover.__module__ = verified_agentic.__name__
    run_agent_with_failover._codna_agent_model_env_support = True  # type: ignore[attr-defined]
    verified_agentic._run_agent_with_failover = run_agent_with_failover


def install_patch_artifact_newline_preservation() -> None:
    """Patch local Algenta patch reads so diff payload CR bytes are preserved."""
    try:
        execution_workflow = _load_repository_module(
            "apps.api_server.services.repository_intelligence_execution_workflow"
        )
        core = _load_repository_module("apps.api_server.services.repository_intelligence_core")
    except Exception as exc:  # noqa: BLE001
        raise LocalRepositoryShimError(
            "local_repository_patch_newline_shim_install_failed",
            "Codna could not install the local patch-artifact newline preservation shim.",
            {"reason": f"{type(exc).__name__}: {exc}"},
        ) from exc

    original_patch_text = getattr(execution_workflow, "_patch_text_for_plan", None)
    if original_patch_text is None:
        raise LocalRepositoryShimError(
            "local_repository_patch_newline_shim_missing_target",
            "Local Algenta execution workflow is missing _patch_text_for_plan.",
            {"module": execution_workflow.__name__},
        )
    if not getattr(original_patch_text, "_codna_preserves_patch_newlines", False):

        def patch_text_for_plan(
            *,
            org_id: str,
            repository_id: str,
            decision_plan_manifest: dict[str, Any],
        ) -> str:
            patch_path_value = decision_plan_manifest.get("patch_path")
            if isinstance(patch_path_value, str):
                patch_path = Path(patch_path_value)
                if patch_path.exists():
                    return read_text_preserving_newlines(patch_path)
            return original_patch_text(
                org_id=org_id,
                repository_id=repository_id,
                decision_plan_manifest=decision_plan_manifest,
            )

        patch_text_for_plan.__name__ = "_patch_text_for_plan"
        patch_text_for_plan.__qualname__ = "_patch_text_for_plan"
        patch_text_for_plan.__module__ = execution_workflow.__name__
        patch_text_for_plan._codna_preserves_patch_newlines = True  # type: ignore[attr-defined]
        execution_workflow._patch_text_for_plan = patch_text_for_plan

    original_digest_loader = getattr(core, "_load_patch_diff_by_digest", None)
    if original_digest_loader is None:
        return
    if getattr(original_digest_loader, "_codna_preserves_patch_newlines", False):
        return

    def load_patch_diff_by_digest(
        *,
        org_id: str,
        repository_id: str,
        patch_digest: str,
    ) -> str | None:
        return _load_patch_diff_by_digest_preserving_newlines(
            core=core,
            org_id=org_id,
            repository_id=repository_id,
            patch_digest=patch_digest,
        )

    load_patch_diff_by_digest.__name__ = "_load_patch_diff_by_digest"
    load_patch_diff_by_digest.__qualname__ = "_load_patch_diff_by_digest"
    load_patch_diff_by_digest.__module__ = core.__name__
    load_patch_diff_by_digest._codna_preserves_patch_newlines = True  # type: ignore[attr-defined]
    core._load_patch_diff_by_digest = load_patch_diff_by_digest


def read_text_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def git_apply_compatible_unified_diff_from_tree_states(
    before: dict[str, str],
    after: dict[str, str],
) -> str:
    segments: list[str] = []
    for relative in sorted(set(before) | set(after)):
        if _is_generated_diff_artifact(relative):
            continue
        old_text = before.get(relative)
        new_text = after.get(relative)
        if old_text == new_text:
            continue
        from_lines = old_text.splitlines(keepends=True) if old_text is not None else []
        to_lines = new_text.splitlines(keepends=True) if new_text is not None else []
        from_file = "/dev/null" if old_text is None else f"a/{relative}"
        to_file = "/dev/null" if new_text is None else f"b/{relative}"
        for line in difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=from_file,
            tofile=to_file,
            lineterm="\n",
        ):
            segments.append(_patch_line_with_explicit_terminator(line))
    return "".join(segments)


def _is_generated_diff_artifact(relative: str) -> bool:
    path = Path(relative)
    if any(part in _GENERATED_DIFF_SKIP_DIRS for part in path.parts):
        return True
    if path.name in _GENERATED_DIFF_SKIP_FILES:
        return True
    return path.name.endswith(_GENERATED_DIFF_SKIP_SUFFIXES)


def precise_patch_changed_files(patch_diff: str) -> dict[str, list[tuple[int, int]]]:
    changed_lines: dict[str, set[int]] = {}
    current_file: str | None = None
    old_file: str | None = None
    old_line: int | None = None
    new_line: int | None = None
    for line in patch_diff.splitlines():
        old_match = _PATCH_OLD_FILE_RE.match(line)
        if old_match:
            old_file = _normalize_patch_path(old_match.group(1))
            continue
        new_match = _PATCH_NEW_FILE_RE.match(line)
        if new_match:
            new_path = _normalize_patch_path(new_match.group(1))
            current_file = old_file if new_path is None else new_path
            old_line = None
            new_line = None
            if current_file is not None:
                changed_lines.setdefault(current_file, set())
            continue
        hunk_match = _PATCH_HUNK_RE.match(line)
        if hunk_match and current_file is not None:
            old_line = int(hunk_match.group("old_start"))
            new_line = int(hunk_match.group("new_start"))
            continue
        if current_file is None or old_line is None or new_line is None:
            continue
        old_line, new_line = _record_patch_line_change(
            changed_lines[current_file],
            line,
            old_line=old_line,
            new_line=new_line,
        )
    return {
        path: _compact_changed_line_ranges(lines)
        for path, lines in changed_lines.items()
        if lines
    }


def _load_patch_diff_by_digest_preserving_newlines(
    *,
    core: ModuleType,
    org_id: str,
    repository_id: str,
    patch_digest: str,
) -> str | None:
    try:
        repository_runtime = _load_repository_module(
            "apps.api_server.services.repository_intelligence_runtime"
        )
    except Exception:  # noqa: BLE001 - matches upstream miss-tolerant digest lookup.
        return None
    try:
        plans_dir = repository_runtime._decision_plans_dir(org_id, repository_id)
        manifests = sorted(plans_dir.glob("*.json"))
    except OSError:
        return None
    for manifest_path in manifests:
        try:
            manifest = repository_runtime._read_json(manifest_path)
        except (OSError, ValueError):
            continue
        diff = manifest.get("patch_diff")
        if not diff and manifest.get("patch_path"):
            try:
                diff = read_text_preserving_newlines(Path(manifest["patch_path"]))
            except OSError:
                diff = None
        if isinstance(diff, str) and core._codna_patch_digest(diff) == patch_digest:
            return diff
    return None


def _patch_line_with_explicit_terminator(line: str) -> str:
    if line.endswith("\n"):
        return line
    normalized = f"{line}\n"
    if _is_unified_diff_payload_line(line):
        return f"{normalized}\\ No newline at end of file\n"
    return normalized


def _is_unified_diff_payload_line(line: str) -> bool:
    if not line:
        return False
    if line.startswith(("--- ", "+++ ", "@@ ")):
        return False
    return line[0] in {" ", "+", "-"}


def _load_repository_module(module_name: str) -> ModuleType:
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    return importlib.import_module(module_name)


def _extend_text_file_suffixes(module: ModuleType, suffixes: frozenset[str]) -> None:
    current = getattr(module, "TEXT_FILE_SUFFIXES", None)
    if current is None:
        raise LocalRepositoryShimError(
            "local_repository_missing_text_suffix_registry",
            "Local repository-intelligence module is missing TEXT_FILE_SUFFIXES.",
            {"module": module.__name__},
        )
    if hasattr(current, "update"):
        current.update(suffixes)
        return
    setattr(module, "TEXT_FILE_SUFFIXES", set(current) | set(suffixes))


def _extend_direct_language_map(classification: ModuleType) -> None:
    direct_map = getattr(classification, "_DIRECT_TEXT_SUFFIX_LANGUAGE_MAP", None)
    if isinstance(direct_map, dict):
        direct_map.update(_LOCAL_SDK_DIRECT_SUFFIX_LANGUAGES)


def _extend_js_like_suffixes(indexing: ModuleType) -> None:
    current = getattr(indexing, "_JS_LIKE_SUFFIXES", ())
    if not isinstance(current, tuple):
        return
    seen = set(current)
    additions = tuple(
        suffix
        for suffix in _LOCAL_SDK_JS_LIKE_SUFFIXES
        if suffix not in seen
    )
    if additions:
        indexing._JS_LIKE_SUFFIXES = (*current, *additions)


def _patch_language_for_path(module: ModuleType) -> None:
    original = getattr(module, "_language_for_path", None)
    if original is None:
        return
    if getattr(original, "_codna_module_suffix_support", False):
        return

    def language_for_path(path: Path, *args: Any, **kwargs: Any) -> str:
        language = _LOCAL_SDK_DIRECT_SUFFIX_LANGUAGES.get(Path(path).suffix.lower())
        if language is not None:
            return language
        return original(path, *args, **kwargs)

    language_for_path.__name__ = getattr(original, "__name__", "_language_for_path")
    language_for_path.__qualname__ = getattr(original, "__qualname__", language_for_path.__name__)
    language_for_path.__doc__ = getattr(original, "__doc__", None)
    language_for_path.__module__ = getattr(original, "__module__", module.__name__)
    language_for_path._codna_module_suffix_support = True  # type: ignore[attr-defined]
    module._language_for_path = language_for_path


def _record_patch_line_change(
    changed_lines: set[int],
    line: str,
    *,
    old_line: int,
    new_line: int,
) -> tuple[int, int]:
    if line.startswith("\\"):
        return old_line, new_line
    if line.startswith("+") and not line.startswith("+++"):
        changed_lines.add(max(new_line, 1))
        return old_line, new_line + 1
    if line.startswith("-") and not line.startswith("---"):
        changed_lines.add(max(new_line, 1))
        return old_line + 1, new_line
    return old_line + 1, new_line + 1


def _normalize_patch_path(path: str) -> str | None:
    normalized = path.strip()
    if normalized == "/dev/null":
        return None
    return normalized


def _compact_changed_line_ranges(lines: set[int]) -> list[tuple[int, int]]:
    sorted_lines = sorted(lines)
    if not sorted_lines:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = sorted_lines[0]
    for line in sorted_lines[1:]:
        if line == previous + 1:
            previous = line
            continue
        ranges.append((start, previous))
        start = previous = line
    ranges.append((start, previous))
    return ranges
