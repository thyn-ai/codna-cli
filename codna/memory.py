"""Codna's local, in-process code memory — a thin consumer seam over Telys.

Telys ships with base ``codna`` as the public SDK plus Codna's signed on-device runtime package
data. ``codna[memory]`` only adds heavier optional retrieval/rerank dependencies; Codna does not depend
on a private runtime wheel in public PyPI metadata. Codna resolves the native kernel from explicit ``$TELYS_KERNEL`` /
``$AME_KERNEL``, a packaged ``codna/_telys_runtime`` artifact, ``$CODNA_TELYS_INSTALL_ROOT``,
``build/local-telys`` in a source checkout, or a verified Telys runtime install. Core codna never imports
it — every telys import here is lazy, so ``from codna.memory import CodeMemory`` works without the extra
and only fails (with a friendly message) when you actually use it.

OEM license handling is explicit and secret-safe. Codna holds a 100-year OEM Telys entitlement that
covers every Codna user, so users never obtain their own Telys license. A release wheel may bake that
license alongside the packaged kernel (``codna/_telys_runtime/oem-license.jwt``, staged at build time,
never committed) so base ``codna`` activates with zero Telys configuration. An explicit
``CODNA_TELYS_LICENSE_JWT`` / ``CODNA_TELYS_LICENSE_PATH`` always overrides the bundled default. Either
way the license is verified offline through Telys and only metadata is retained — the JWT is never
returned or logged. Set ``CODNA_TELYS_LICENSE_REQUIRED=1`` in channels that must fail closed.

The embedder is resolved in :func:`_embedder`: when an engine URL + API key are configured, recall uses
the **API-key-gated engine** ``/v1/embeddings`` for REAL semantic vectors (the model auto-tracks the LLM
used to fix the code — ``ALGENTA_AGENT_PROVIDER`` — overridable via ``CODNA_MEMORY_EMBED_MODEL``). With no
engine/key (or ``CODNA_MEMORY_EMBED=local``) it falls back to the in-process ``AlgentaMultigramEmbedder``
stand-in (384-d unigram+bigram+trigram fusion, lexical, zero network — the MC-validated upgrade over the
old bigram default). Switching embedders changes the ``space_id`` → re-index required.

Precision rerank (ON by default): the top-k pool is reordered by a lexical-heavy BLEND of the lexical
recall score and an on-device WordLlama semantic score (``alpha*lexical + (1-alpha)*wordllama``, alpha=0.7
— the best-measured config). Tune/disable via ``CODNA_MEMORY_RERANK`` (``blend`` default | ``wordllama`` |
``lexical``/``off``) and ``CODNA_MEMORY_RERANK_ALPHA``. WordLlama ships in the ``[memory]`` extra; if it's
somehow absent the blend falls back to lexical (never hard-fails).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from .codeunits import FILTER_COLUMNS, UnsupportedLanguageError, extract_repo, resolve_languages
from .memory_manifest import legacy_manifest_id_belongs_to_repo, manifest_entry, manifest_hash, manifest_scope

COLLECTION = "repo_symbols"
DEFAULT_DB_DIRNAME = ".codna-memory"
META_FILENAME = "codna_memory.json"
HASH_FILENAME = "content_hashes.json"   # local id->content-hash manifest for incremental re-index (Merkle-style)
_LICENSE_ENV_NAMES = ("CODNA_TELYS_LICENSE_JWT", "TELYS_LICENSE_JWT")
_LICENSE_PATH_ENV_NAMES = ("CODNA_TELYS_LICENSE_PATH", "TELYS_LICENSE_PATH")
_LICENSE_REQUIRED_ENV = "CODNA_TELYS_LICENSE_REQUIRED"
_KERNEL_ENV_NAMES = ("TELYS_KERNEL", "AME_KERNEL")
_CODNA_KERNEL_ENV_NAME = "CODNA_TELYS_KERNEL"
_CODNA_INSTALL_ROOT_ENV_NAME = "CODNA_TELYS_INSTALL_ROOT"
_PACKAGE_RUNTIME_DIRNAME = "_telys_runtime"
_PACKAGE_RUNTIME_MANIFEST = "manifest.json"
_PACKAGE_RUNTIME_SCHEMA_VERSION = 1
# Codna holds a 100-year OEM Telys entitlement covering every Codna user, so users never obtain their
# own Telys license. When a release bakes that OEM license here (a companion to the packaged kernel,
# staged at build time — never committed), base codna activates with zero user configuration.
_PACKAGE_LICENSE_FILENAME = "oem-license.jwt"
_LAST_KERNEL_RESOLUTION: dict | None = None


class CodeMemoryError(Exception):
    """A codna-memory error surfaced to the user (missing extra / kernel / usage).

    Deliberately NOT named ``MemoryError`` — that is a Python builtin and shadowing it would make
    ``except MemoryError`` ambiguous. The CLI maps this to ``_die`` and the MCP tool to an error string.
    """


def _kernel_filename() -> str:
    if sys.platform == "darwin":
        return "libame_kernel.dylib"
    if sys.platform.startswith("win"):
        return "libame_kernel.dll"
    return "libame_kernel.so"


def _source_checkout_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_platform_tag() -> str:
    machine = platform.machine().lower() or "unknown"
    if sys.platform == "darwin":
        return f"macos-{machine}"
    if sys.platform.startswith("linux"):
        return f"linux-{machine}"
    if sys.platform.startswith("win"):
        return f"windows-{machine}"
    return f"{sys.platform}-{machine}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_runtime_dir() -> Path:
    return Path(__file__).resolve().parent / _PACKAGE_RUNTIME_DIRNAME


def _load_package_runtime_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CodeMemoryError(f"Codna packaged Telys runtime manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CodeMemoryError(f"Codna packaged Telys runtime manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CodeMemoryError(f"Codna packaged Telys runtime manifest must be a JSON object: {path}")
    return payload


def _validate_package_runtime_manifest(runtime_dir: Path, manifest: dict) -> dict:
    expected_artifact = _kernel_filename()
    if manifest.get("schema_version") != _PACKAGE_RUNTIME_SCHEMA_VERSION:
        raise CodeMemoryError("Codna packaged Telys runtime manifest has unsupported schema_version")
    if manifest.get("artifact") != expected_artifact:
        raise CodeMemoryError(
            f"Codna packaged Telys runtime artifact must be {expected_artifact}; "
            f"manifest has {manifest.get('artifact')!r}"
        )
    if manifest.get("platform") != _runtime_platform_tag():
        raise CodeMemoryError(
            f"Codna packaged Telys runtime platform mismatch: {manifest.get('platform')!r}"
        )
    for key in ("sha256", "size_bytes", "source_basename"):
        if key not in manifest:
            raise CodeMemoryError(f"Codna packaged Telys runtime manifest is missing {key}")
    if "source_path" in manifest:
        raise CodeMemoryError("Codna packaged Telys runtime manifest must not store source_path")
    artifact = runtime_dir / expected_artifact
    if not artifact.is_file():
        raise CodeMemoryError(f"Codna packaged Telys runtime kernel is missing: {artifact}")
    if artifact.stat().st_size != manifest["size_bytes"]:
        raise CodeMemoryError("Codna packaged Telys runtime kernel size does not match manifest")
    if _sha256_file(artifact) != manifest["sha256"]:
        raise CodeMemoryError("Codna packaged Telys runtime kernel sha256 does not match manifest")
    return {
        "manifest_path": str((runtime_dir / _PACKAGE_RUNTIME_MANIFEST).resolve()),
        "sha256": manifest["sha256"],
        "size_bytes": manifest["size_bytes"],
        "platform": manifest["platform"],
    }


def _package_runtime_candidate() -> tuple[Path, dict] | None:
    runtime_dir = _package_runtime_dir()
    manifest_path = runtime_dir / _PACKAGE_RUNTIME_MANIFEST
    kernels = sorted(runtime_dir.glob("libame_kernel.*")) if runtime_dir.is_dir() else []
    if not manifest_path.exists() and not kernels:
        return None
    expected = _kernel_filename()
    unexpected = [path.name for path in kernels if path.name != expected]
    if unexpected:
        raise CodeMemoryError(
            "Codna packaged Telys runtime contains unsupported kernel artifact(s): "
            + ", ".join(unexpected)
        )
    manifest = _load_package_runtime_manifest(manifest_path)
    metadata = _validate_package_runtime_manifest(runtime_dir, manifest)
    return runtime_dir / expected, metadata


def _candidate_codna_install_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    raw = os.environ.get(_CODNA_INSTALL_ROOT_ENV_NAME)
    if raw and raw.strip():
        roots.append((f"env:{_CODNA_INSTALL_ROOT_ENV_NAME}", Path(raw).expanduser()))
    roots.append(("codna:source-build", _source_checkout_root() / "build" / "local-telys"))
    return roots


def _candidate_kernel_paths() -> list[tuple[str, Path]]:
    filename = _kernel_filename()
    candidates: list[tuple[str, Path]] = []
    raw_kernel = os.environ.get(_CODNA_KERNEL_ENV_NAME)
    if raw_kernel and raw_kernel.strip():
        candidates.append((f"env:{_CODNA_KERNEL_ENV_NAME}", Path(raw_kernel).expanduser()))
    packaged = _package_runtime_candidate()
    if packaged is not None:
        candidates.append(("codna:package-runtime", packaged[0]))
    for source, root in _candidate_codna_install_roots():
        candidates.append((source, root / "kernel" / filename))
    try:
        from telys.paths import AME_KERNEL_STEM, installed_lib_path

        installed = installed_lib_path(AME_KERNEL_STEM)
        if installed:
            candidates.append(("telys:installed-runtime", Path(installed)))
    except Exception:  # noqa: BLE001 - kernel diagnostics must not mask the real runtime import error.
        pass
    return candidates


def _explicit_kernel_env() -> tuple[str, str] | None:
    for name in _KERNEL_ENV_NAMES:
        if name in os.environ:
            return name, os.environ.get(name, "")
    return None


def _resolve_telys_kernel(*, configure_env: bool) -> dict:
    explicit = _explicit_kernel_env()
    if explicit is not None:
        name, raw = explicit
        if not raw.strip():
            raise CodeMemoryError(f"{name} is set but empty")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise CodeMemoryError(f"Telys kernel path from {name} does not exist or is not a file: {path}")
        return {"found": True, "source": f"env:{name}", "path": str(path)}

    checked: list[str] = []
    for source, raw_path in _candidate_kernel_paths():
        path = raw_path.expanduser().resolve()
        checked.append(f"{source}:{path}")
        if path.is_file():
            if configure_env:
                os.environ["TELYS_KERNEL"] = str(path)
            result = {"found": True, "source": source, "path": str(path)}
            if source == "codna:package-runtime":
                _, metadata = _package_runtime_candidate() or (path, {})
                result.update(metadata)
            return result
    return {"found": False, "source": "missing", "path": None, "checked": checked}


def _configure_telys_kernel_env() -> dict:
    global _LAST_KERNEL_RESOLUTION
    _LAST_KERNEL_RESOLUTION = _resolve_telys_kernel(configure_env=True)
    return dict(_LAST_KERNEL_RESOLUTION)


def _current_kernel_resolution() -> dict:
    if _LAST_KERNEL_RESOLUTION is not None:
        return dict(_LAST_KERNEL_RESOLUTION)
    return _resolve_telys_kernel(configure_env=False)


def _kernel_hint() -> str:
    resolution = _current_kernel_resolution()
    checked = resolution.get("checked") or []
    checked_lines = "\n".join(f"  - {item}" for item in checked[:6])
    if checked_lines:
        checked_lines = "\nCodna checked:\n" + checked_lines
    return (
        "the Telys kernel could not be loaded. Run `codna login` to authorize this device and "
        "provision the on-device runtime (free for one device), install a Codna platform wheel with "
        "the bundled runtime, set TELYS_KERNEL=/abs/path/to/libame_kernel, or set CODNA_TELYS_INSTALL_ROOT to a Codna-managed "
        f"install containing kernel/{_kernel_filename()}.{checked_lines}"
    )


def _runtime_install_hint() -> str:
    return (
        "the Telys runtime is not installed. Reinstall or upgrade `codna` from a platform wheel that "
        "bundles the signed runtime, then run `codna login` to authorize this device if a per-device "
        "login license is needed. For source checkout or unsupported-platform development, use a "
        "Codna-packaged runtime artifact or set TELYS_KERNEL/CODNA_TELYS_INSTALL_ROOT."
    )


def _require_telys():
    """Lazily import the Telys surface. Raises CodeMemoryError with an install hint if absent."""
    try:
        from telys import Telys, scope_key
    except ImportError as exc:
        raise CodeMemoryError(
            "code memory needs the Telys SDK shipped with base `codna`; reinstall or upgrade `codna` "
            "from a platform wheel, or activate a Codna-packaged signed runtime"
        ) from exc
    _configure_telys_kernel_env()
    try:
        # Multigram fusion ([unigram|bigram|trigram], per-block L2): MC-validated to beat the plain bigram
        # by ~+12-17 pts recall@10 on NL->code and to tie a 16 MB on-device semantic model on recall, at
        # zero model cost. The local lexical default (no engine key) uses it.
        from telys.embedding import AlgentaMultigramEmbedder as _LocalEmbedder
    except ImportError as exc:
        raise CodeMemoryError(_kernel_hint()) from exc
    return Telys, scope_key, _LocalEmbedder


def _sdk_supports_lexical(eng) -> bool:
    """True when the resolved Telys SDK exposes the lexical (BM25) lane — ``create_collection``
    accepts ``lexical=`` and collections gain ``build_lexical()`` / ``search_text(mode='lexical')``.
    Detected by signature, never assumed: an older Telys keeps dense-only recall (fail open)."""
    try:
        import inspect
        sig = inspect.signature(eng.create_collection)
        return "lexical" in sig.parameters
    except (TypeError, ValueError):  # pragma: no cover — defensive; signature introspection failed
        return False


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _package_license_path() -> Path:
    """Path to the OEM license baked into the release wheel (may not exist)."""
    return _package_runtime_dir() / _PACKAGE_LICENSE_FILENAME


def _onboarding_license_path() -> Path:
    """Path where `codna login` (telys.login) writes this device's per-device offline license.

    Mirrors telys.paths.telys_home() ($TELYS_HOME or ~/.telys) without importing telys.
    """
    home = os.path.abspath(os.environ.get("TELYS_HOME") or os.path.join(os.path.expanduser("~"), ".telys"))
    return Path(home) / "login_license.jwt"


def _configured_license_token() -> tuple[str | None, str | None]:
    for name in _LICENSE_ENV_NAMES:
        raw = os.environ.get(name)
        if raw is None:
            continue
        token = raw.strip()
        if not token:
            raise CodeMemoryError(f"{name} is set but empty")
        return token, f"env:{name}"

    for name in _LICENSE_PATH_ENV_NAMES:
        raw = os.environ.get(name)
        if raw is None:
            continue
        if not raw.strip():
            raise CodeMemoryError(f"{name} is set but empty")
        path = os.path.abspath(os.path.expanduser(raw.strip()))
        try:
            with open(path, encoding="utf-8") as fh:
                token = fh.read().strip()
        except OSError as exc:
            raise CodeMemoryError(f"Telys license path from {name} is unreadable: {path}") from exc
        if not token:
            raise CodeMemoryError(f"Telys license path from {name} is empty: {path}")
        return token, f"path:{name}"

    # Per-device license from `codna login` (telys.login device-code onboarding). This is the primary
    # entitlement for the standard product: each device authorizes and gets its own device-scoped
    # license. It wins over the embedded OEM umbrella below so per-device licensing is the default.
    onboarding = _onboarding_license_path()
    if onboarding.is_file():
        try:
            token = onboarding.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CodeMemoryError(f"Codna device login license is unreadable: {onboarding}") from exc
        if token:
            return token, "codna:onboarding-license"

    # Codna's embedded OEM entitlement: a release wheel may bake the 100-year license alongside the
    # kernel so every Codna user is covered without configuring or obtaining their own Telys license.
    # An explicit env/path (above) always wins; this is the zero-config default.
    bundled = _package_license_path()
    if bundled.is_file():
        try:
            token = bundled.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CodeMemoryError(
                f"Codna packaged Telys OEM license is unreadable: {bundled}"
            ) from exc
        if not token:
            raise CodeMemoryError(f"Codna packaged Telys OEM license is empty: {bundled}")
        return token, "codna:package-license"

    return None, None


def _load_telys_license_metadata() -> dict:
    """Verify the configured Telys license offline and return safe metadata only.

    This deliberately does not return the JWT. Callers can surface the metadata in status/diagnostics without
    risking token disclosure.
    """
    required = _truthy_env(_LICENSE_REQUIRED_ENV)
    token, source = _configured_license_token()
    if token is None:
        if required:
            raise CodeMemoryError(
                "Telys license is required but not configured; set CODNA_TELYS_LICENSE_JWT "
                "or CODNA_TELYS_LICENSE_PATH"
            )
        return {"configured": False, "required": False}
    try:
        from telys import verify as license_verify
    except ImportError as exc:
        raise CodeMemoryError(
            "Telys license is configured but the Telys verifier is unavailable; reinstall or upgrade "
            "codna from a platform wheel"
        ) from exc
    try:
        claims = license_verify.verify_license(token, now=int(time.time()))
    except Exception as exc:  # noqa: BLE001
        raise CodeMemoryError(f"Telys license from {source} is invalid: {type(exc).__name__}") from exc

    product = (claims.get("products") or {}).get("telys") or {}
    if not product:
        raise CodeMemoryError(f"Telys license from {source} does not include the telys product")
    return {
        "configured": True,
        "required": required,
        "source": source,
        "license_id": claims.get("license_id"),
        "customer": claims.get("customer"),
        "org_id": claims.get("org_id"),
        "deployment_class": claims.get("deployment_class"),
        "tier": product.get("tier"),
        "features": list(product.get("features") or []),
    }


# Map the fix LLM provider → its registered embedding-model id. Chat models (claude/gpt) can't embed, so
# we use the SAME provider's embedding model. Anthropic ships no embeddings API → use OpenAI's (whose key
# is already configured for the gpt-5.4 fix path). Override per run with CODNA_MEMORY_EMBED_MODEL.
_PROVIDER_EMBED_MODEL = {
    "anthropic": "codna.embed.openai",
    "openai-native": "codna.embed.openai", "openai": "codna.embed.openai",
    "gemini": "codna.embed.gemini", "google_genai": "codna.embed.gemini",
}
_EMBED_DIM = {"codna.embed.openai": 1024, "codna.embed.gemini": 1536}


def _engine_embed_config():
    """Resolve (engine_url, api_key, model_id, dimension) for the API-key-gated remote code embedder,
    or ``None`` to fall back to the local stand-in. The model auto-tracks the fix provider
    (``ALGENTA_AGENT_PROVIDER``); override with ``CODNA_MEMORY_EMBED_MODEL`` / ``_DIM``. Force the
    in-process stand-in with ``CODNA_MEMORY_EMBED=local``; require the remote with ``=remote``."""
    if (os.environ.get("CODNA_MEMORY_EMBED") or "").lower() == "local":
        return None
    url = os.environ.get("CODNA_ENGINE_URL") or os.environ.get("ALGENTA_ENGINE_URL")
    key = (os.environ.get("CODNA_API_KEY") or os.environ.get("ALGENTA_API_KEY")
           or os.environ.get("ALGENTA_ENGINE_API_KEY"))
    if not (url and key):
        return None
    model = os.environ.get("CODNA_MEMORY_EMBED_MODEL")
    if not model:
        prov = (os.environ.get("ALGENTA_AGENT_PROVIDER") or "anthropic").lower()
        model = _PROVIDER_EMBED_MODEL.get(prov, "codna.embed.openai")
    dim = int(os.environ.get("CODNA_MEMORY_EMBED_DIM") or _EMBED_DIM.get(model, 1024))
    return url.rstrip("/"), key, model, dim


def _remote_embedder(url: str, key: str, model: str, dim: int):
    """A ``CallableEmbedder`` backed by the engine's keyed ``/v1/embeddings`` — REAL semantic vectors
    (the "get it from Telys/Algenta with the API key" path). L2-normalized so inner-product == cosine."""
    import httpx
    import numpy as np
    from telys.embedding import CallableEmbedder, EmbeddingProfile

    def _embed(texts):
        out = []
        for i in range(0, len(texts), 128):  # bounded request bodies
            chunk = list(texts[i:i + 128])
            body = {"input": chunk, "model": model, "dimensions": dim}
            # Resilient to TRANSIENT blips (connect/timeout, 429, 5xx): a momentary embedding-endpoint
            # hiccup must NOT fail an otherwise-good recall/fix. Retry with backoff; only a persistent
            # failure (or a real 4xx) propagates — so `--memory on` still fails loud when memory is truly
            # broken, but survives a one-off network/engine stutter (the websockets soak case).
            for attempt in range(4):
                try:
                    r = httpx.post(f"{url}/v1/embeddings",
                                   headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
                except httpx.TransportError:
                    if attempt == 3:
                        raise
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                r.raise_for_status()
                out.extend(item["embedding"] for item in sorted(r.json()["data"], key=lambda d: d["index"]))
                break
        v = np.asarray(out, dtype="float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)
    return CallableEmbedder(
        fn=_embed,
        profile=EmbeddingProfile(provider="algenta-engine", model_id=model, model_version="1",
                                 dimension=dim, normalization="l2", distance="ip", tokenizer_hash=model))


def _embedder():
    """Resolve the code embedder. Prefer the API-key-gated engine embedder (real semantic recall,
    auto-tracking the fix provider); fall back to the in-process multigram fusion stand-in (lexical,
    on-device, no key) — the MC-validated upgrade over the old bigram default."""
    cfg = _engine_embed_config()
    if cfg is not None:
        try:
            return _remote_embedder(*cfg)
        except Exception as exc:  # noqa: BLE001 — telys import / httpx / engine down
            if (os.environ.get("CODNA_MEMORY_EMBED") or "").lower() == "remote":
                raise CodeMemoryError(f"remote embedder required (CODNA_MEMORY_EMBED=remote) "
                                      f"but unavailable: {exc}") from exc
            # else: silently fall through to the local stand-in below
    _, _, LocalEmbedder = _require_telys()
    try:
        return LocalEmbedder()    # AlgentaMultigramEmbedder: unigram+bigram+trigram fusion, zero network
    except (OSError, ImportError) as exc:   # FileNotFoundError / CDLL load / missing runtime kernel module
        raise CodeMemoryError(_kernel_hint()) from exc


# Recall-time rerank pipeline — split into memory_rerank.py (MOD-FILESIZE 1000-line ceiling). The
# names are re-imported here so the public import surface (`from codna.memory import ...`) and the
# recall() call sites below are unchanged.
from .memory_rerank import (  # noqa: E402, F401
    _is_test_symbol,
    _query_is_test_oriented,
    _semantic_rerank,
    _soft_test_downweight,
    _wl_cosine,
    _wordllama,
    rerank,
)


def repo_id_for(repo_path: str) -> str:
    """Stable repo identity for partitioning: ``github.com/owner/repo`` if there's an origin remote,
    else the absolute path (so memory works on any local checkout, not only GitHub repos).

    Git deliberately walks up parent directories. That is correct for commands run at a repo root, but
    unsafe for arbitrary local paths: a temporary non-git repo under this source checkout must not inherit
    Codna's own GitHub remote and collide with other paths in a shared collection.
    """
    local = os.path.abspath(os.path.expanduser(repo_path))
    try:
        root = subprocess.run(
            ["git", "-C", local, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_root = os.path.abspath(os.path.expanduser(root.stdout.strip())) if root.returncode == 0 else ""
        if git_root and os.path.realpath(git_root) == os.path.realpath(local):
            out = subprocess.run(["git", "-C", local, "remote", "get-url", "origin"],
                                 capture_output=True, text=True, timeout=5)
        else:
            out = None
        if out is not None and out.returncode == 0:
            m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", out.stdout)
            if m:
                return f"github.com/{m.group(1)}"
    except (OSError, subprocess.SubprocessError):
        pass
    return local


def _git(repo_path: str, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", repo_path, *args],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _provenance(unit_id: str) -> dict:
    """Recover path + qualname from a stable unit id (``repo_id:relpath:qualname``)."""
    parts = unit_id.split(":")
    if len(parts) >= 3:
        return {"path": parts[-2], "qualname": parts[-1]}
    return {"path": None, "qualname": parts[-1] if parts else unit_id}


def _metadata_matches(metadata: dict, filters: dict[str, str]) -> bool:
    return all(str(metadata.get(key) or "") == str(value) for key, value in filters.items())


class CodeMemory:
    """In-process semantic code memory for one repository, backed by a Telys collection."""

    def __init__(self, repo_path: str, db_path: str | None = None, *,
                 service: str | None = None, branch: str | None = None, tuner=None) -> None:
        self.repo_path = os.path.abspath(os.path.expanduser(repo_path))
        self.repo_id = repo_id_for(self.repo_path)
        self.db_path = os.path.abspath(db_path) if db_path else os.path.join(self.repo_path, DEFAULT_DB_DIRNAME)
        self.service = service
        self.branch = branch
        self._tuner = tuner
        self._Telys, self._scope_key, _ = _require_telys()
        self.license = _load_telys_license_metadata()
        self.kernel = _current_kernel_resolution()
        self._embedder = _embedder()                       # constructs (and validates) the kernel up-front
        try:
            self._eng = self._Telys(
                self.db_path,
                embedding_providers={self._embedder.profile.model_id: self._embedder},
            )
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "RuntimeNotInstalled":
                raise CodeMemoryError(_runtime_install_hint()) from exc
            raise
        self._col = None

    # ── collection lifecycle ──────────────────────────────────────────────────────────────────────
    def _collection(self):
        if self._col is None:
            lexical_sdk = _sdk_supports_lexical(self._eng)
            if COLLECTION in self._eng.collections():
                self._col = self._eng.open_collection(COLLECTION, embedder=self._embedder)
                if (lexical_sdk and not getattr(self._col, "lexical", False)
                        and not getattr(self, "_lexical_migrated", False)):
                    # The lexical (BM25) lane needs per-row tokens retained at INGEST time — an index
                    # built before the hybrid-recall seam cannot grow them in place. One transparent
                    # rebuild upgrades it; guarded so it happens at most once per instance.
                    self._lexical_migrated = True
                    self.reset()
                    self._col = None
                    return self._collection()
            else:
                kwargs = dict(dim=self._embedder.profile.dimension, partition_by="scope_key",
                              embedder=self._embedder, filter_columns=FILTER_COLUMNS, tuner=self._tuner,
                              dtype=getattr(self, "_dtype_override", None) or "f32")
                if lexical_sdk:
                    kwargs["lexical"] = True
                self._col = self._eng.create_collection(COLLECTION, **kwargs)
            self._lexical = bool(getattr(self._col, "lexical", False))
        return self._col

    def is_empty(self) -> bool:
        if COLLECTION not in self._eng.collections():
            return True
        return (self._collection().stats().get("external_ids") or 0) == 0

    def _scope_for(self, unit) -> str:
        return self._scope_key(self.repo_id, self.service or unit.service or "", unit.language)

    def _unit_texts(self, languages=None) -> dict:
        """{unit_id: source_text} for this repo, cached. Used ONLY by the optional WordLlama reranker to get
        candidate text at query time, so the index stores no extra text column.

        Normally the cache is populated by ``index()`` (reusing its extraction, all indexed languages). If
        recall runs on a fresh instance without a prior ``index()``, fall back to extracting the languages
        last indexed (persisted intent) or all registered languages — never silently Python-only when the
        repo is multi-language."""
        cache = getattr(self, "_unit_text_cache", None)
        if cache is None:
            langs = languages or getattr(self, "_indexed_languages", None)
            units, _ = extract_repo(self.repo_path, self.repo_id, languages=langs)
            cache = {u.id: u.text for u in units}
            self._unit_text_cache = cache
        return cache

    # ── index / recall / status / compact / reset ─────────────────────────────────────────────────
    def index(self, *, languages: tuple[str, ...] | None = None, refresh: bool = True) -> dict:
        """Extract the repo into structural units and upsert them per scope; delete vanished symbols.

        The vanished-diff uses Codna's local content-hash manifest as the prune authority. Telys search is
        still the retrieval engine, but prune must be deterministic and scope-local even if the runtime's
        id enumeration semantics change. Re-indexing one repo/scope never touches another's rows in a
        shared collection.
        """
        try:
            resolved_languages = resolve_languages(languages)
            units, stats = extract_repo(self.repo_path, self.repo_id, languages=resolved_languages)
        except UnsupportedLanguageError as exc:
            raise CodeMemoryError(str(exc)) from exc
        col = self._collection()
        # Reuse this full extraction as the WordLlama reranker's candidate-text cache: multi-language
        # (every indexed language, not just Python) AND no re-extraction at query time. Without this the
        # reranker would only see Python text and score every non-Python candidate as an empty string.
        self._indexed_languages = resolved_languages
        self._unit_text_cache = {u.id: u.text for u in units}
        commit = _git(self.repo_path, "rev-parse", "HEAD") or ""
        branch = self.branch or _git(self.repo_path, "rev-parse", "--abbrev-ref", "HEAD") or ""

        # Incremental re-index: skip embedding symbols whose content is unchanged since the last index.
        # The per-symbol hash is salted by the embedder's space_id, so a model change invalidates every entry
        # (a fresh embed). Manifest is local (no engine round-trip); delete it to force a full re-embed.
        salt = self._embedder.profile.space_id()
        def _uhash(u):
            service_key = self.service or u.service or ""
            return hashlib.sha256((salt + "\x1f" + service_key + "\x1f" + u.text).encode("utf-8")).hexdigest()[:16]
        prev = self._load_hashes() if refresh else {}
        new_hashes: dict[str, str] = {}

        by_scope: dict[str, list] = {}
        new_ids: dict[str, set] = {}
        for u in units:
            scope = self._scope_for(u)
            by_scope.setdefault(scope, []).append(u)
            new_ids.setdefault(scope, set()).add(u.id)

        removed = 0
        vanished: list[str] = []
        if refresh:
            # The scopes this index owns ((repo_id, service) × the scopes written + the requested
            # languages, so an emptied scope is still pruned). Delete only ids from those scopes that the
            # current extraction no longer produces. Older string-only manifests are supported by falling
            # back to the stable repo-id prefix.
            owned = set(new_ids) | {self._scope_key(self.repo_id, self.service or "", lang)
                                    for lang in resolved_languages}
            produced = set().union(*new_ids.values()) if new_ids else set()
            vanished = sorted(
                unit_id
                for unit_id, entry in prev.items()
                if unit_id not in produced
                and (
                    manifest_scope(entry) in owned
                    or (manifest_scope(entry) is None and legacy_manifest_id_belongs_to_repo(unit_id, self.repo_id))
                )
            )
            if vanished:
                col.delete(vanished)
                removed = len(vanished)

        written = unchanged = 0
        new_hashes: dict[str, dict] = {}
        for scope, group in by_scope.items():
            changed = []
            for u in group:
                h = _uhash(u)
                service_key = self.service or u.service or ""
                new_hashes[u.id] = manifest_entry(
                    digest=h,
                    repo_id=self.repo_id,
                    service=service_key,
                    language=u.language,
                    scope_key=scope,
                )
                if manifest_hash(prev.get(u.id)) == h:        # unchanged content (same space) -> skip the embed/upsert
                    unchanged += 1
                else:
                    changed.append(u)
            if changed:
                metadata = []
                for u in changed:
                    item = u.metadata(self.repo_id, scope, commit_sha=commit, branch=branch)
                    item["service"] = self.service or u.service or ""
                    metadata.append(item)
                col.upsert_texts([u.text for u in changed], [u.id for u in changed],
                                 metadata)
                written += len(changed)

        col.compact()
        self._build_lexical(col)
        col.save()
        merged_hashes = dict(prev) if refresh else {}
        for unit_id in vanished:
            merged_hashes.pop(unit_id, None)
        merged_hashes.update(new_hashes)
        self._save_hashes(merged_hashes)
        self._write_meta(commit=commit, branch=branch)
        return {"indexed": written, "unchanged": unchanged, "removed": removed, "scopes": len(by_scope),
                "files": stats.files, "skipped": stats.skipped, "languages": list(resolved_languages)}

    def _build_lexical(self, col) -> None:
        """Fit the on-device BM25 lane over the freshly indexed rows (compact() invalidated any prior
        layout). Fail OPEN to dense-only recall where the runtime has no lexical support — the native
        runtime raises NotImplementedError today; hybrid switches on automatically once the lane ships."""
        if not getattr(self, "_lexical", False):
            return
        try:
            col.build_lexical()
        except NotImplementedError:
            self._lexical = False

    def _hash_path(self) -> str:
        return os.path.join(self.db_path, HASH_FILENAME)

    def _load_hashes(self) -> dict:
        try:
            with open(self._hash_path()) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_hashes(self, m: dict) -> None:
        os.makedirs(self.db_path, exist_ok=True)
        tmp = self._hash_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f)
        os.replace(tmp, self._hash_path())

    def current_commit(self) -> str:
        return _git(self.repo_path, "rev-parse", "HEAD") or ""

    def is_stale(self) -> bool:
        """True when the repo HEAD moved since the last index, so recall would serve OLD symbols.
        Conservative: only reports stale when BOTH commits are known and differ (a non-git repo or a
        missing meta never forces a re-index)."""
        indexed = (self._read_meta().get("commit_sha") or "")
        cur = self.current_commit()
        return bool(cur) and bool(indexed) and cur != indexed

    def embedder_changed(self) -> bool:
        """True when the active embedder's vector SPACE differs from the indexed one (different embedding
        model / dim / normalization — e.g. the fix provider switched and auto-selected a new embed model).
        Vectors across spaces are incompatible, so the index must be rebuilt, not reopened."""
        prev = self._read_meta().get("space_id")
        return bool(prev) and prev != self._embedder.profile.space_id()

    def ensure_fresh(self, *, languages: tuple[str, ...] | None = None) -> dict | None:
        """Keep recall honest before use. Rebuilds the index when the EMBEDDER changed (incompatible
        vector space → must re-embed everything), else refreshes when the repo HEAD moved (add new,
        re-embed changed, prune vanished). No-op (returns ``None``) when both are unchanged. This is what
        makes memory robust across LLM/embedder switches — switching the fix LLM with a pinned embedder is
        a no-op; switching the embedder transparently re-indexes instead of erroring on a space mismatch."""
        if self.embedder_changed():
            self.reset()                       # drop the incompatible-space index, then rebuild fresh
            return self.index(languages=languages, refresh=True)
        if self.is_stale():
            return self.index(languages=languages, refresh=True)
        return None

    def recall(self, query: str, *, service: str | None = None, language: str | None = None,
               path: str | None = None, top_k: int = 40, final_k: int = 8,
               target_recall: float = 0.98) -> dict:
        """Partition-aware retrieve (dense + on-device BM25 hybrid when the lexical lane is live) ->
        consumer rerank -> final-k symbols with provenance + explain."""
        col = self._collection()
        active_service = service if service is not None else self.service
        post_filters: dict[str, str] = {}
        search_where: dict[str, str] | None = None
        if language is not None:
            search_where = {"scope_key": self._scope_key(self.repo_id, active_service or "", language)}
            if path is not None:
                post_filters["path"] = path
        elif path is not None:
            search_where = {"path": path}
            if active_service is not None:
                post_filters["service"] = active_service
        elif active_service is not None:
            search_where = {"service": active_service}
        res = col.search_text(query, top_k=top_k, where=search_where, explain=True,
                              target_recall=target_recall, with_metadata=True)
        md = res.get("metadata") or [{}] * len(res["ids"])   # provenance returned WITH the hit (Telys)
        meta_by_id = dict(zip(res["ids"], md))
        dense_scores = dict(zip(res["ids"], res["scores"]))
        if getattr(self, "_lexical", False):
            # HYBRID: fuse the on-device BM25 lane with the dense lane. Dense misses bare identifiers
            # (cosine dilution — a 60-char qualname query vs a 2000-char unit buries the exact match
            # ~rank 1900); lexical matches exact tokens. The POOL is unioned by reciprocal-rank
            # fusion (dense cosines and BM25 scores live on incomparable scales, ranks don't; k=60
            # is the standard RRF constant), but the score handed to the rerank blend is the
            # candidate's LANE score — BM25 where the lexical lane ranked it (the blend's lexical
            # component, alpha 0.7, was tuned on BM25-scale inputs), dense cosine otherwise.
            try:
                lex = col.search_text(query, top_k=top_k, where=search_where, mode="lexical",
                                      with_metadata=True)
                lex_md = lex.get("metadata") or [{}] * len(lex["ids"])
                lex_scores: dict[str, float] = {}
                for unit_id, m, score in zip(lex["ids"], lex_md, lex["scores"]):
                    lex_scores[unit_id] = score
                    if unit_id not in meta_by_id:
                        meta_by_id[unit_id] = m or {}
                k_rrf = 60
                fused: dict[str, float] = {}
                for rank, unit_id in enumerate(res["ids"]):
                    fused[unit_id] = fused.get(unit_id, 0.0) + 1.0 / (k_rrf + rank + 1)
                for rank, unit_id in enumerate(lex["ids"]):
                    fused[unit_id] = fused.get(unit_id, 0.0) + 1.0 / (k_rrf + rank + 1)
                order = sorted(fused, key=fused.__getitem__, reverse=True)
                scores_by_id = {uid: lex_scores.get(uid, dense_scores.get(uid, 0.0)) for uid in order}
            except (NotImplementedError, RuntimeError):
                order = list(dense_scores)                  # no lexical lane (older runtime) -> dense
                scores_by_id = dense_scores
        else:
            order = list(dense_scores)
            scores_by_id = dense_scores
        # Lossless query-aware soft rerank over the FULL candidate set, then the (future) rerank seam,
        # then final-k — so a source symbol buried under tests can surface without dropping any coverage.
        raw_cands = [
            (unit_id, scores_by_id[unit_id])
            for unit_id in order
            if _metadata_matches(meta_by_id.get(unit_id) or {}, post_filters)
        ]
        cands = _soft_test_downweight(query, raw_cands, meta_by_id)
        cands = _semantic_rerank(query, cands, self)    # DEFAULT: lexical+WordLlama blend (alpha 0.7); falls back
        ranked = rerank(query, cands)[:final_k]          # to lexical if WordLlama absent. CODNA_MEMORY_RERANK to tune.
        symbols = []
        for i, s in ranked:
            m = meta_by_id.get(i) or {}
            symbols.append({"id": i, "score": s, "symbol_type": m.get("symbol_type"),
                            "path": m.get("path") or _provenance(i)["path"]})
        return {"symbols": symbols, "explain": res.get("explain", {}),
                "candidate_count": len(raw_cands), "final_k": len(symbols)}

    def status(self) -> dict:
        meta = self._read_meta()
        if COLLECTION not in self._eng.collections():
            return {"engine": "telys", "collection": COLLECTION, "indexed": False,
                    "db_path": self.db_path, "repo_id": self.repo_id, "license": self.license,
                    "kernel": self.kernel}
        s = self._collection().stats()
        return {
            "engine": "telys",
            "collection": COLLECTION,
            "indexed": True,
            "lexical": bool(getattr(self, "_lexical", False)),   # hybrid BM25 lane live for recall?
            "partition_key": s.get("key_name", "scope_key"),
            "partitions": s.get("partitions"),
            "documents": s.get("external_ids"),
            "delta_rows": s.get("delta_rows"),
            "embedding_space": s.get("embedding_space") or meta.get("space_id"),
            "last_indexed_commit": meta.get("commit_sha"),
            "last_indexed_at": meta.get("indexed_at"),
            "fallback_rate": None,                         # no query log yet (Telys exports no metrics)
            "db_path": self.db_path,
            "repo_id": self.repo_id,
            "license": self.license,
            "kernel": self.kernel,
        }

    def compact(self) -> dict:
        col = self._collection()
        col.compact()
        self._build_lexical(col)          # compact() invalidates the per-partition lexical layouts
        col.save()
        return {"compacted": True, "documents": col.stats().get("external_ids")}

    def export_serve_artifact(self, path: str, mode: str = "int8") -> dict:
        """Export the index as an ultra-small READ-ONLY serve artifact (telys compact tier): quantized
        slab + remap + provenance columns — no f32 base, no id map, no lexical tokens. Makes the
        memory current first (index when empty, refresh when stale), then seals. ``mode``: "int8"
        (~4x smaller — default) or "pq" (~32x scan working set; needs faiss). Requires a telys with
        the compact seam (telys#99); older telys raises upgrade guidance, never a bad artifact.

        The compact tiers keep the int8 slab, which only exists when the collection was sealed at
        dtype="int8". If this memory's collection is f32, a ONE-TIME transparent reset+reindex at
        int8 runs first — search quality is tier-validated (int8 == f32 recall on the quant PoC)."""
        col = self._collection()
        export = getattr(col, "export_compact", None)
        if export is None:
            raise CodeMemoryError(
                "the resolved telys predates compact serve artifacts (telys#99) — upgrade telys")
        if getattr(col, "dtype", "f32") != "int8":
            self.reset()
            self._col = None
            self._dtype_override = "int8"
            self.index()
            col = self._collection()
            # Re-bind: the bound method captured above still points at the pre-rebuild collection.
            export = getattr(col, "export_compact", None)
        if self.is_empty():
            self.index()
        else:
            self.ensure_fresh()
        out = export(path, mode=mode)
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
        return {"artifact": out, "mode": mode, "documents": col.stats().get("external_ids"),
                "size_bytes": size, "size_mb": round(size / 1e6, 2)}

    def reset(self) -> dict:
        """Drop the local memory for this repo (collection + meta + incremental hash manifest). Safe no-op
        if nothing is indexed.

        The hash manifest MUST be cleared too: it drives the incremental re-index (#56), so a stale manifest
        after dropping the collection makes the next index() treat every symbol as "unchanged" and skip
        embedding it — leaving an empty collection (recall returns nothing). reset() must be a full reset."""
        self._col = None
        existed = os.path.isdir(os.path.join(self.db_path, COLLECTION))
        shutil.rmtree(os.path.join(self.db_path, COLLECTION), ignore_errors=True)
        for fn in (META_FILENAME, HASH_FILENAME, HASH_FILENAME + ".tmp"):
            p = os.path.join(self.db_path, fn)
            if os.path.exists(p):
                os.remove(p)
        return {"reset": True, "existed": existed, "db_path": self.db_path}

    # ── side metadata (commit/branch/symbol-types) — Codna-side, separate from telys' collection.json ─
    def _meta_path(self) -> str:
        return os.path.join(self.db_path, META_FILENAME)

    def _read_meta(self) -> dict:
        try:
            with open(self._meta_path()) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _write_meta(self, *, commit: str, branch: str) -> None:
        os.makedirs(self.db_path, exist_ok=True)
        # Just the index-run record for `status`. NO per-id map: provenance now comes from
        # search(with_metadata=True) and the vanished-diff from col.ids(where=scope) (Telys 0.1.0-alpha).
        payload = {
            "commit_sha": commit,
            "branch": branch,
            "indexed_at": int(time.time()),
            "space_id": self._embedder.profile.space_id(),
        }
        with open(self._meta_path(), "w") as f:
            json.dump(payload, f)
