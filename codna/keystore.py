"""On-device provider-key store for Codna (local BYOK), backed by the OS keychain.

Local Codna keeps the user's LLM provider key (Anthropic/OpenAI/…) — and optionally the Codna engine
key — ON THE DEVICE. It never travels to the cloud. This is the local counterpart to the website BYOK
card (Supabase-authed, encrypted server-side): same "add your key" idea, storage chosen by where the
model call runs. Here the call runs locally, so the key lives in the OS secure store (macOS Keychain,
Windows Credential Manager, Linux Secret Service) via ``keyring`` — not a plaintext ``.env``.

Resolution (see ``cli._runtime_keys`` + ``runtime.config``): an explicit env var always wins, then
Codna-indexed keychain entries, then a legacy source-checkout ``keys.txt``. The value is write-only
from the CLI (``codna key set`` reads it via a no-echo prompt / stdin, never argv) and is never
printed back.

Runtime discovery reads a local non-secret key-name index first. If Codna has never stored key names
on this device, discovery does not probe the OS keychain just to find absence. Explicit writes/removes
still surface :class:`KeystoreError` with an actionable hint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .runtime.config import ConfigValue

SERVICE = "codna"
# Index entry: keyring backends can't enumerate stored items, so we keep our own name index under a
# reserved username. (No secret lives in the index — only key NAMES.)
_INDEX_USERNAME = "__codna_key_index__"
_LOCAL_INDEX_NAME = "keychain-index.json"

# Provider/LLM keys the local engine forwards to the fixing agent (mirror of
# runtime.local_stack.PROVIDER_CHILD_ENV_KEYS) + the Codna engine key. Only these may be stored.
MANAGED_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
    "CURSOR_API_KEY",
    "CODNA_API_KEY",
)

# Friendly aliases so `codna key set anthropic` works (in addition to the raw env-var name).
PROVIDER_ALIASES: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
    "cursor": "CURSOR_API_KEY",
    "codna": "CODNA_API_KEY",
    "engine": "CODNA_API_KEY",
}


class KeystoreError(RuntimeError):
    """A keychain operation the user asked for could not be completed (surfaced with remediation)."""


def disabled() -> bool:
    """True when keychain access is explicitly disabled for headless/test execution."""
    return os.environ.get("CODNA_DISABLE_KEYCHAIN", "").strip().lower() in {"1", "true", "yes", "on"}


def _keyring():
    """Import keyring or raise KeystoreError with an install hint. Never imported at module load."""
    if disabled():
        raise KeystoreError("OS keychain access is disabled by CODNA_DISABLE_KEYCHAIN=1")
    try:
        import keyring
    except Exception as exc:  # noqa: BLE001 - any import failure means no usable keychain
        raise KeystoreError(
            "the OS keychain is unavailable — install it with `pip install keyring` (bundled with "
            "codna), or set the provider key as an environment variable instead"
        ) from exc
    return keyring


def normalize_key_name(name: str) -> str:
    """Map a provider alias or env-var name to a managed env-var key name (or raise)."""
    raw = (name or "").strip()
    if not raw:
        raise KeystoreError("a provider name or *_API_KEY variable name is required")
    alias = PROVIDER_ALIASES.get(raw.lower())
    resolved = alias or raw.upper()
    if resolved not in MANAGED_KEYS:
        managed = ", ".join(sorted(set(PROVIDER_ALIASES) | set(MANAGED_KEYS)))
        raise KeystoreError(f"unknown provider key '{name}'. Known: {managed}")
    return resolved


def available() -> bool:
    """True if a usable OS keychain backend is present (best-effort, never raises)."""
    if disabled():
        return False
    try:
        kr = _keyring()
        backend = getattr(kr, "get_keyring", lambda: None)()
        priority = getattr(backend, "priority", 0)
        return priority is None or float(priority) > 0
    except Exception:  # noqa: BLE001 - KeystoreError (no keyring) or a backend that refuses reads
        return False


def _state_root() -> Path:
    override = os.environ.get("CODNA_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codna"


def _local_index_path() -> Path:
    return _state_root() / "keys" / _LOCAL_INDEX_NAME


def _normalize_index_names(names: list[str]) -> list[str]:
    return sorted({name for name in names if name in MANAGED_KEYS})


def _read_local_index() -> list[str] | None:
    path = _local_index_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - malformed local metadata must not trigger keychain probes
        return []
    names = payload.get("keys") if isinstance(payload, dict) else None
    if not isinstance(names, list):
        return []
    return _normalize_index_names([str(name) for name in names])


def _write_local_index(names: list[str]) -> None:
    path = _local_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"schema_version": 1, "keys": _normalize_index_names(names)}
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _read_keychain_index() -> list[str]:
    kr = _keyring()
    raw = kr.get_password(SERVICE, _INDEX_USERNAME)
    # str() defends against a non-conformant backend returning a non-string (keyring's contract is
    # Optional[str], but `codna key list/set/rm` must not traceback on a misbehaving third-party store).
    names = [n.strip() for n in str(raw or "").split(",") if n.strip()]
    return _normalize_index_names(names)


def _write_keychain_index(names: list[str]) -> None:
    kr = _keyring()
    kr.set_password(SERVICE, _INDEX_USERNAME, ",".join(_normalize_index_names(names)))


def _read_index() -> list[str]:
    local = _read_local_index()
    if local is not None:
        return local
    # No local non-secret index means Codna has not explicitly stored key names on this device.
    # Do not probe the OS keychain just to discover absence; that causes repeated prompts.
    return []


def _read_index_for_explicit_write() -> list[str]:
    local = _read_local_index()
    if local is not None:
        return local
    try:
        return _read_keychain_index()
    except Exception:  # noqa: BLE001
        return []


def _write_index(names: list[str]) -> None:
    unique = _normalize_index_names(names)
    _write_local_index(unique)
    _write_keychain_index(unique)


def set_key(name: str, value: str) -> str:
    """Store a provider key in the OS keychain. Returns the resolved env-var name."""
    key = normalize_key_name(name)
    token = (value or "").strip()
    if not token:
        raise KeystoreError("refusing to store an empty key")
    kr = _keyring()
    try:
        kr.set_password(SERVICE, key, token)
    except Exception as exc:  # noqa: BLE001 - backend write failure
        raise KeystoreError(f"could not write to the OS keychain: {type(exc).__name__}") from exc
    index = _read_index_for_explicit_write()
    if key not in index:
        _write_index([*index, key])
    return key


def get_key(name: str) -> str | None:
    """Read a stored provider key, or None if absent / no keychain. Never raises."""
    try:
        key = normalize_key_name(name)
        kr = _keyring()
        value = kr.get_password(SERVICE, key)
    except Exception:  # noqa: BLE001
        return None
    return value.strip() if value and value.strip() else None


def delete_key(name: str) -> bool:
    """Remove a stored provider key. Returns True if one existed. Idempotent."""
    key = normalize_key_name(name)
    kr = _keyring()
    existed = False
    try:
        if kr.get_password(SERVICE, key):
            existed = True  # observed present BEFORE delete, so a swallowed PasswordDeleteError can't reset it
            kr.delete_password(SERVICE, key)
    except Exception as exc:  # noqa: BLE001
        # delete_password raises PasswordDeleteError when absent on some backends — treat as "gone".
        from keyring.errors import PasswordDeleteError  # type: ignore

        if not isinstance(exc, PasswordDeleteError):
            raise KeystoreError(f"could not delete from the OS keychain: {type(exc).__name__}") from exc
    index = _read_index()
    if key in index:
        _write_index([n for n in index if n != key])
    return existed


def stored_key_names() -> list[str]:
    """Names in the key index (values never read or returned). Best-effort; never raises."""
    return _read_index()


def config_values() -> dict[str, ConfigValue]:
    """Keychain-sourced keys as ConfigValues for the runtime key map. Never raises (returns {})."""
    out: dict[str, ConfigValue] = {}
    try:
        for name in _read_index():
            value = get_key(name)
            if value:
                out[name] = ConfigValue(key=name, value=value, source="keychain")
    except Exception:  # noqa: BLE001 - a broken keychain must never break the fix hot path
        return {}
    return out
