from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence
from urllib.parse import urlparse

from .. import __version__

DEFAULT_PORT_BASE = 18600
ENGINE_READY_TIMEOUT_S = 60.0
HEALTH_TIMEOUT_S = 2.0
LOCK_TIMEOUT_S = 30.0
LOG_ROTATE_BYTES = 15 * 1024 * 1024
LOG_ROTATE_FILES = 2
READY_TIMEOUT_S = 5.0
RUNTIME_OWNER = "codna-local-runtime"
SCHEMA_VERSION = 1
SIDECAR_READY_TIMEOUT_S = 45.0
STOP_FORCE_TIMEOUT_S = 5.0
STOP_GRACE_TIMEOUT_S = 10.0

ENGINE_URL_KEYS = ("CODNA_ENGINE_URL", "ALGENTA_ENGINE_URL", "ALGENTA_BASE_URL")
# The ONLY way an engine checkout is named. There is deliberately no default and no home-directory guess:
# `~/Developer/decision-engine` was the implicit default, which on a user's machine either does not exist
# (silent degradation) or is an unrelated project that happens to own that name — it is what activated the
# legacy Mojo pool daemon from a probe. A path that genuinely needs the checkout refuses with
# `engine_dir_unset` and names this variable (see `require_engine_dir`).
ENGINE_DIR_ENV_VAR = "ALGENTA_ENGINE_DIR"
API_KEY_KEYS = ("CODNA_API_KEY", "ALGENTA_API_KEY", "DE_API_KEY")
LOCAL_DATABASE_ENV_KEYS = ("DATABASE_URL", "ASYNC_DATABASE_URL")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}
FINGERPRINT_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    "node_modules",
}


class RuntimeConfigError(RuntimeError):
    def __init__(self, message: str, *, code: str = "runtime_config_error", details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class ConfigValue:
    key: str
    value: str
    source: Literal["env", "keys_txt", "keychain", "default"]


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    runtime_dir: Path
    logs_dir: Path
    state_path: Path
    lock_path: Path
    migration_marker_path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    codna_home: Path
    port_base: int
    engine_port: int
    sidecar_port: int
    engine_url: str
    sidecar_url: str
    remote_engine_url: str | None
    ignored_engine_url: ConfigValue | None
    engine_url_override: ConfigValue | None
    paths: RuntimePaths
    # ``None`` unless ALGENTA_ENGINE_DIR names a checkout. Everything that only needs the sidecar or a
    # remote engine works with ``None``; the few paths that genuinely need the local engine call
    # ``require_engine_dir`` and fail with a coded refusal naming the variable rather than guessing.
    engine_dir: Path | None
    engine_python: str
    engine_env_file: Path
    sidecar_dir: Path
    sidecar_binary: Path | None
    cline_data_dir: Path
    codna_cli_version: str
    engine_build_id: str
    sidecar_build_id: str
    runtime_config_hash: str
    allow_loopback_override: bool


def _codna_home() -> Path:
    return Path(__file__).resolve().parents[3]


def _codna_package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_root() -> Path:
    override = os.environ.get("CODNA_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".codna"


def build_runtime_paths() -> RuntimePaths:
    root = _runtime_root()
    runtime_dir = root / "runtime"
    logs_dir = root / "logs"
    return RuntimePaths(
        root=root,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        state_path=runtime_dir / "local-stack.json",
        lock_path=runtime_dir / "local-stack.lock",
        migration_marker_path=runtime_dir / "legacy-cleanup-v1.done",
    )


def parse_keys_file_values(path: Path) -> dict[str, ConfigValue]:
    parsed: dict[str, ConfigValue] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = ConfigValue(
            key=key.strip(),
            value=value.strip(),
            source="keys_txt",
        )
    return parsed


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def url_host(url: str) -> str | None:
    return urlparse(url).hostname


def is_loopback_url(url: str) -> bool:
    host = url_host(url)
    return bool(host and host in LOOPBACK_HOSTS)


def _config_value(
    name: str,
    *,
    env: Mapping[str, str],
    keys: Mapping[str, ConfigValue] | None,
) -> ConfigValue | None:
    raw_env = env.get(name)
    if raw_env:
        return ConfigValue(key=name, value=raw_env, source="env")
    if keys:
        return keys.get(name)
    return None


def _first_config_value(
    names: tuple[str, ...],
    *,
    env: Mapping[str, str],
    keys: Mapping[str, ConfigValue] | None,
) -> ConfigValue | None:
    for name in names:
        value = _config_value(name, env=env, keys=keys)
        if value and value.value:
            return value
    return None


def _validate_port_base(raw: str | None) -> int:
    if raw is None or raw == "":
        return DEFAULT_PORT_BASE
    try:
        base = int(raw)
    except ValueError as exc:
        raise RuntimeConfigError(
            "CODNA_PORT_BASE must be an integer.",
            code="invalid_port_base",
            details={"value": raw},
        ) from exc
    if base < 1024 or base > 65534:
        raise RuntimeConfigError(
            "CODNA_PORT_BASE must be between 1024 and 65534.",
            code="invalid_port_base",
            details={"value": base},
        )
    return base


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            if key.isidentifier():
                env[key] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def local_engine_defaults() -> dict[str, str]:
    return {
        "ALGENTA_DEPLOYMENT_MODE": "self_hosted",
        "ALGENTA_DISABLE_CLOUD": "1",
        "ALGENTA_DISABLE_TELEMETRY": "1",
        "ALGENTA_TELEMETRY_MODE": "local_audit",
        "ALGENTA_METERING_MODE": "local_audit",
        "ALGENTA_REQUIRE_PER_TENANT_KEYS": "0",
        "ALGENTA_MANAGED_LLM": "0",
    }


def _engine_dir_from_env(resolved_env: Mapping[str, str]) -> Path | None:
    """The engine checkout the environment names, or ``None``.

    NO DEFAULT, deliberately. The previous implementation fell back to ``Path.home()/Developer/decision-engine``;
    that guess is the defect: on a user's machine the directory either does not exist (so the failure surfaced
    far from its cause) or exists as an unrelated project with that name (so Codna imported a stranger's
    ``apps.*`` package and started a Mojo daemon against it). Absence is now a first-class state, and the paths
    that cannot proceed without a checkout call :func:`require_engine_dir`, which names the variable to set.
    """
    raw = (resolved_env.get(ENGINE_DIR_ENV_VAR) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def engine_dir_unset_error(*, consumer: str, details: dict | None = None) -> RuntimeConfigError:
    """The structured refusal for a path that genuinely needs an engine checkout."""
    return RuntimeConfigError(
        f"{consumer} needs a local Algenta engine checkout, and {ENGINE_DIR_ENV_VAR} is not set.",
        code="engine_dir_unset",
        details={
            "required_env": ENGINE_DIR_ENV_VAR,
            "consumer": consumer,
            "expected_contents": "a checkout containing apps/api_server/main.py",
            **dict(details or {}),
        },
    )


def require_engine_dir(config: RuntimeConfig, *, consumer: str, details: dict | None = None) -> Path:
    """The resolved engine directory, or a coded refusal that names what to set.

    Every path that reads files out of the checkout (the FastAPI engine spawn, the in-process import path, the
    Mojo pool, the worker-state report) goes through here so the refusal is one shape everywhere. Callers must
    NOT substitute a default: a guessed path is what this function exists to remove.
    """
    if config.engine_dir is None:
        raise engine_dir_unset_error(consumer=consumer, details=details)
    return config.engine_dir


def _agent_core_root_from_override(raw: str) -> Path:
    path = Path(raw).expanduser()
    if (path / "run-server.mjs").is_file():
        return path
    if (path / "algenta" / "run-server.ts").is_file() and path.parent.name == "vendor":
        return path.parent.parent
    return path


def _default_sidecar_dir() -> Path:
    checkout_candidate = _codna_home() / "agent-core"
    package_candidate = _codna_package_root() / "agent-core"
    candidates = (checkout_candidate, package_candidate)
    for candidate in candidates:
        if (candidate / "run-server.mjs").is_file():
            return candidate
    return package_candidate


# Self-contained agent-core: a `bun build --compile` standalone that embeds the Bun runtime + the
# sidecar server + SDK into ONE per-platform binary, staged into the wheel next to the Telys native
# libs (`codna/_agent_core_runtime/`). When present it is launched directly — no Node, no Bun, no
# node_modules — so a bare `pip install codna` runs `codna fix` with zero JS-runtime setup. A source
# checkout (no compiled binary) transparently falls back to the Node supervisor + Bun dev path.
SIDECAR_RUNTIME_DIRNAME = "_agent_core_runtime"


def _sidecar_binary_name() -> str:
    return "codna-sidecar.exe" if os.name == "nt" else "codna-sidecar"


def _default_sidecar_binary() -> Path | None:
    override = os.environ.get("CODNA_SIDECAR_BINARY")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    name = _sidecar_binary_name()
    for base in (
        _codna_package_root() / SIDECAR_RUNTIME_DIRNAME,
        _codna_home() / SIDECAR_RUNTIME_DIRNAME,
    ):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _file_fingerprint(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_file():
        hasher.update(path.read_bytes())
    else:
        hasher.update(str(path).encode("utf-8"))
    return hasher.hexdigest()


def _update_hasher_for_path(
    hasher: hashlib._Hash,
    path: Path,
    *,
    label: str,
    suffixes: Sequence[str] | None = None,
) -> None:
    if not path.exists():
        hasher.update(f"missing:{label}".encode("utf-8"))
        hasher.update(b"\0")
        return
    if path.is_file():
        if suffixes and path.suffix not in suffixes:
            return
        hasher.update(label.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
        return
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative_child = child.relative_to(path)
        if any(part in FINGERPRINT_IGNORED_DIRS for part in relative_child.parts):
            continue
        if suffixes and child.suffix not in suffixes:
            continue
        hasher.update(f"{label}/{relative_child.as_posix()}".encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(child.read_bytes())
        hasher.update(b"\0")


def _composite_fingerprint(entries: Sequence[tuple[Path, str, Sequence[str] | None]]) -> str:
    hasher = hashlib.sha256()
    for path, label, suffixes in entries:
        _update_hasher_for_path(hasher, path, label=label, suffixes=suffixes)
    return hasher.hexdigest()


def _runtime_config_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_runtime_config(
    *,
    env: Mapping[str, str] | None = None,
    keys: Mapping[str, ConfigValue] | None = None,
) -> RuntimeConfig:
    resolved_env = env or os.environ
    port_base = _validate_port_base(resolved_env.get("CODNA_PORT_BASE"))
    engine_port = port_base
    sidecar_port = port_base + 1
    allow_loopback_override = resolved_env.get("CODNA_ALLOW_LOOPBACK_ENGINE_URL") == "1"
    engine_override = _first_config_value(ENGINE_URL_KEYS, env=resolved_env, keys=None)
    ignored_engine_url = _first_config_value(ENGINE_URL_KEYS, env={}, keys=keys)
    remote_engine_url: str | None = None
    if engine_override:
        normalized_url = normalize_url(engine_override.value)
        if is_loopback_url(normalized_url) and not allow_loopback_override:
            raise RuntimeConfigError(
                "Explicit loopback engine URLs are disabled; unset CODNA_ENGINE_URL or set "
                "CODNA_ALLOW_LOOPBACK_ENGINE_URL=1 for dev-only overrides.",
                code="loopback_override_rejected",
                details={"url": normalized_url, "source": engine_override.source},
            )
        remote_engine_url = normalized_url
    engine_url = f"http://127.0.0.1:{engine_port}"
    sidecar_url = f"http://127.0.0.1:{sidecar_port}"
    codna_home = _codna_home()
    paths = build_runtime_paths()
    # NO DEFAULT. `None` means "no checkout was named", which is a valid configuration for every path that
    # only needs the sidecar or a remote engine. The paths that read the checkout call `require_engine_dir`.
    engine_dir = _engine_dir_from_env(resolved_env)
    engine_python = resolved_env.get("ALGENTA_ENGINE_PYTHON") or (
        str(engine_dir / ".venv" / "bin" / "python") if engine_dir is not None else sys.executable
    )
    if not Path(engine_python).exists():
        engine_python = sys.executable
    # With no checkout there is no checkout `.env` to read; the owned runtime root is the only place left that
    # Codna itself controls, and `load_env_file` treats a missing file as empty.
    engine_env_file = Path(
        resolved_env.get("ALGENTA_ENGINE_ENV_FILE")
        or ((engine_dir / ".env") if engine_dir is not None else (paths.root / "engine.env"))
    ).expanduser()
    sidecar_override = resolved_env.get("CODNA_SIDECAR_DIR") or resolved_env.get("CODNA_AGENT_CORE_DIR")
    sidecar_dir = (
        _agent_core_root_from_override(sidecar_override)
        if sidecar_override
        else _default_sidecar_dir()
    )
    # Prefer the bundled self-contained sidecar binary (no Node/Bun). An explicit source-dir override
    # (CODNA_SIDECAR_DIR / CODNA_AGENT_CORE_DIR) forces the Node+Bun dev path unless the caller also
    # points CODNA_SIDECAR_BINARY at a specific binary.
    if sidecar_override and not os.environ.get("CODNA_SIDECAR_BINARY"):
        sidecar_binary: Path | None = None
    else:
        sidecar_binary = _default_sidecar_binary()
    cline_data_dir = Path(
        resolved_env.get("CLINE_DATA_DIR")
        or (paths.root / "sidecar-state" / "cline-data")
    ).expanduser()
    # The engine build id is a fingerprint of the checkout. With no checkout there is nothing to fingerprint,
    # so it is the empty-string sentinel rather than a hash of a path that does not exist — a hash of a
    # guessed path is exactly the "looks configured" state this change removes.
    if engine_dir is None:
        engine_build_id = ""
    else:
        engine_build_path = engine_dir / "apps" / "api_server" / "main.py"
        engine_build_id = (
            _file_fingerprint(engine_build_path) if engine_build_path.exists() else _file_fingerprint(engine_dir)
        )
    _sidecar_fingerprint_targets: list[tuple[Path, str, tuple[str, ...] | None]] = []
    if sidecar_binary is not None:
        _sidecar_fingerprint_targets.append((sidecar_binary, "sidecar-binary", None))
    _sidecar_fingerprint_targets += [
        (sidecar_dir / "run-server.mjs", "run-server.mjs", None),
        (sidecar_dir / "package.json", "package.json", None),
        (
            sidecar_dir / "vendor" / "cline" / "algenta",
            "vendor/cline/algenta",
            (".ts", ".js", ".mjs", ".json"),
        ),
        (
            sidecar_dir / "vendor" / "cline" / "sdk" / "packages" / "core",
            "vendor/cline/sdk/packages/core",
            (".ts", ".js", ".mjs", ".json"),
        ),
    ]
    sidecar_build_id = _composite_fingerprint(_sidecar_fingerprint_targets)
    config_hash = _runtime_config_hash(
        {
            "port_base": port_base,
            # `None` and a real path must hash differently, and the string "None" must not collide with a
            # directory literally named `None`; the sentinel makes the unset state explicit in the hash.
            "engine_dir": str(engine_dir) if engine_dir is not None else None,
            "engine_python": engine_python,
            "engine_env_file": str(engine_env_file),
            "sidecar_dir": str(sidecar_dir),
            "cline_data_dir": str(cline_data_dir),
            "runtime_root": str(paths.root),
            "local_defaults": local_engine_defaults(),
        }
    )
    return RuntimeConfig(
        codna_home=codna_home,
        port_base=port_base,
        engine_port=engine_port,
        sidecar_port=sidecar_port,
        engine_url=engine_url,
        sidecar_url=sidecar_url,
        remote_engine_url=remote_engine_url,
        ignored_engine_url=ignored_engine_url,
        engine_url_override=engine_override,
        paths=paths,
        engine_dir=engine_dir,
        engine_python=engine_python,
        engine_env_file=engine_env_file,
        sidecar_dir=sidecar_dir,
        sidecar_binary=sidecar_binary,
        cline_data_dir=cline_data_dir,
        codna_cli_version=__version__,
        engine_build_id=engine_build_id,
        sidecar_build_id=sidecar_build_id,
        runtime_config_hash=config_hash,
        allow_loopback_override=allow_loopback_override,
    )
