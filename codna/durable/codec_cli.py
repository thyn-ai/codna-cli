"""`codna durable codec {keygen,status,retire}` — the payload-codec key lifecycle.

Kept in its own module for two reasons: `cli.py` and `run_cli.py` are both under the repository's
1000-line ceiling (`test_modularity.py`), and key lifecycle is a distinct operation from running a
durable task.

## What each command does, and what it refuses to do

* ``keygen`` — generate a fresh AES-256 key and store it in the OS keychain as the CURRENT codec key.
  Refuses to overwrite an existing current key without ``--force``; a silent overwrite would make
  every existing payload undecryptable, which is exactly the rotation hazard the codec documents.

* ``status`` — report which mode a client would use, the current key id, the retired key ids, and
  whether the keychain is reachable. Never prints a key VALUE (the returned structures are built from
  `key_id`, a hash prefix, and names only).

* ``retire`` — DEMOTE the current key to a retired entry so a new one can be generated without
  orphaning history. This is the step that makes rotation safe:
  ``keygen --force`` alone is NOT enough; without ``retire`` the old key is gone and old payloads
  become unreadable (loudly, with `payload_key_unavailable`).

A rotation is therefore three explicit steps, which cannot be collapsed into one command by accident:

    codna durable codec retire          # current key -> CODNA_TEMPORAL_CODEC_KEY__<id>
    codna durable codec keygen          # a new current key
    # old history still decodes (its key id is in the ring); new payloads use the new key

There is deliberately no ``rotate`` shortcut. Each step has an observable effect on what can be read,
and an operator who runs only the middle one should see it happen rather than lose history silently.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ..keystore import (
    CODEC_KEY_NAME,
    CODEC_RETIRED_PREFIX,
    KeystoreError,
    delete_key,
    get_key,
    set_key,
    stored_key_names,
)
from .codec import (
    CodecKeyError,
    decode_key_from_keystore,
    encode_key_for_keystore,
    generate_key_material,
    key_id_for,
)

__all__ = ["cmd_durable_codec", "register"]


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _error(message: str, *, code: str, details: dict[str, Any] | None = None) -> Exception:
    from .cli import DurableCliError

    return DurableCliError(message, code=code, details=details or {})


def _current_material():
    raw = get_key(CODEC_KEY_NAME)
    if not raw:
        return None
    try:
        return decode_key_from_keystore(raw)
    except CodecKeyError as exc:
        raise _error(
            "the stored codec key is malformed; generate a new one with `codna durable codec keygen`",
            code=exc.code,
            details=exc.details,
        ) from exc


def _retired() -> list[tuple[str, str]]:
    """`(entry_name, key_id)` for every retired entry, from the non-secret name index."""
    return [
        (name, name[len(CODEC_RETIRED_PREFIX) :])
        for name in stored_key_names()
        if name.startswith(CODEC_RETIRED_PREFIX)
    ]


def cmd_durable_codec(args: Any) -> int:
    """Dispatch on `durable_action=codec` + `codec_action`."""
    action = getattr(args, "codec_action", None)
    if action == "keygen":
        return _keygen(force=bool(getattr(args, "force", False)))
    if action == "status":
        return _status()
    if action == "retire":
        return _retire()
    raise _error(
        "codna durable codec needs a subcommand",
        code="codec_action_required",
        details={"known": ["keygen", "status", "retire"]},
    )


def _keygen(*, force: bool) -> int:
    existing = _current_material()
    if existing is not None and not force:
        raise _error(
            f"a codec key is already stored (id {existing.key_id}); refusing to overwrite it",
            code="codec_key_exists",
            details={
                "key_id": existing.key_id,
                "hint": (
                    "overwriting would make existing history undecryptable; run "
                    "`codna durable codec retire` first, then keygen"
                ),
            },
        )
    material = generate_key_material()
    try:
        set_key(CODEC_KEY_NAME, encode_key_for_keystore(material))
    except KeystoreError as exc:
        raise _error(
            f"could not store the codec key: {exc}",
            code="codec_key_store_failed",
            details={"keystore_entry": CODEC_KEY_NAME},
        ) from exc
    # The VALUE is never printed; the id is a hash prefix and is safe to report.
    _print(
        {
            "ok": True,
            "stored": CODEC_KEY_NAME,
            "key_id": material.key_id,
            "algorithm": "AES-256-GCM",
            "note": "the key value is in the OS keychain and is never printed",
        }
    )
    return 0


def _status() -> int:
    material = _current_material()
    retired = _retired()
    # Resolve the mode the SAME way a client does, so `status` cannot disagree with what runs.
    from .codec_config import CODEC_ENV, resolve_codec

    try:
        resolution = resolve_codec()
        mode_error = None
    except Exception as exc:  # noqa: BLE001 - reported, never a crash for a status command
        resolution = None
        mode_error = {"code": getattr(exc, "code", "codec_mode_invalid"), "message": str(exc)}
    _print(
        {
            "ok": mode_error is None,
            "mode_requested": os.environ.get(CODEC_ENV, "auto"),
            "mode": resolution.mode if resolution else None,
            "reason": resolution.reason if resolution else None,
            "key_id": material.key_id if material else None,
            "retired_key_ids": [key_id for _name, key_id in retired],
            # The entries ACTUALLY present in the non-secret name index, not a constant list — a
            # status command that named an entry that does not exist would be worse than useless.
            "keychain_entries": [
                name
                for name in stored_key_names()
                if name == CODEC_KEY_NAME or name.startswith(CODEC_RETIRED_PREFIX)
            ],
            "error": mode_error,
        }
    )
    return 0 if mode_error is None else 1


def _retire() -> int:
    """Move the current key to a retired entry, so a subsequent `keygen` does not orphan history."""
    material = _current_material()
    if material is None:
        raise _error(
            "there is no current codec key to retire",
            code="codec_key_missing",
            details={"keystore_entry": CODEC_KEY_NAME},
        )
    retired_name = f"{CODEC_RETIRED_PREFIX}{material.key_id}"
    existing = get_key(retired_name)
    if existing and key_id_for(decode_key_from_keystore(existing).key) == material.key_id:
        # Idempotent: the key is already retained under its own id. Do not duplicate it.
        _print({"ok": True, "retired": retired_name, "key_id": material.key_id, "already": True})
        return 0
    raw = get_key(CODEC_KEY_NAME)
    if not raw:  # pragma: no cover - racing delete; the outcome is the same refusal
        raise _error("the current codec key vanished while retiring it", code="codec_key_missing")
    try:
        set_key(retired_name, raw)
    except KeystoreError as exc:
        raise _error(
            f"could not store the retired codec key: {exc}",
            code="codec_key_store_failed",
            details={"keystore_entry": retired_name},
        ) from exc
    # Remove the current entry only AFTER the retired copy is safely stored: the reverse order would
    # leave a window where the key exists nowhere and history is unreadable.
    delete_key(CODEC_KEY_NAME)
    _print(
        {
            "ok": True,
            "retired": retired_name,
            "key_id": material.key_id,
            "next": "run `codna durable codec keygen` to install a new current key",
        }
    )
    return 0


def register(actions: Any) -> None:
    """Attach `codec {keygen,status,retire}` under `codna durable`."""
    from .cli import cmd_durable

    codec = actions.add_parser(
        "codec",
        help="Manage the Temporal payload-encryption key (AES-256-GCM, OS keychain).",
        description=(
            "Temporal history is encrypted with AES-256-GCM. To rotate, run `retire` then `keygen`: "
            "retiring keeps the old key so existing history still decodes."
        ),
    )
    codec_actions = codec.add_subparsers(dest="codec_action")
    keygen = codec_actions.add_parser("keygen", help="Generate and store a new codec key.")
    keygen.add_argument(
        "--force",
        action="store_true",
        help="replace an existing key (existing history becomes undecryptable unless retired first)",
    )
    keygen.set_defaults(func=cmd_durable, durable_action="codec", codec_action="keygen")
    status = codec_actions.add_parser("status", help="Report the codec mode and key ids (never a key).")
    status.set_defaults(func=cmd_durable, durable_action="codec", codec_action="status")
    retire = codec_actions.add_parser(
        "retire", help="Keep the current key for old history, so `keygen` can install a new one."
    )
    retire.set_defaults(func=cmd_durable, durable_action="codec", codec_action="retire")
    codec.set_defaults(func=cmd_durable, durable_action="codec", codec_action="status")
