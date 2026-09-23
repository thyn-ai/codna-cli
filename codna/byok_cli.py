"""CLI glue for local BYOK — the on-device LLM provider key used by local `codna fix`.

Thin layer over :mod:`codna.keystore`: the ``codna key {set,list,rm}`` command and the provider-key
step folded into ``codna login``. The key stays on the device (OS keychain) and is never printed; the
value is read via a no-echo prompt or ``--stdin`` (never argv, to avoid shell-history leaks). Split out
of ``cli.py`` to keep that module under the modularity ceiling.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping

from . import keystore

# LLM provider keys the local fixing agent needs (the engine key CODNA_API_KEY is not a model key).
_PROVIDER_KEYS = tuple(k for k in keystore.MANAGED_KEYS if k != "CODNA_API_KEY")


def provider_key_present(runtime_keys: Mapping[str, object]) -> bool:
    """True if a NON-EMPTY LLM provider key resolves from env or the runtime key map.

    Checks the resolved value, not mere key presence: a value-less ``keys.txt`` line (``ANTHROPIC_API_KEY=``)
    is dropped by the engine's child-env builder, so treating it as "present" would skip onboarding and
    leave the fixing agent with no usable key.
    """
    import os

    if any((os.environ.get(name) or "").strip() for name in _PROVIDER_KEYS):
        return True
    return any(getattr(runtime_keys.get(name), "value", "").strip() for name in _PROVIDER_KEYS)


def ensure_local_provider_key(*, interactive: bool, runtime_keys: Mapping[str, object]) -> str:
    """Offer to store a provider key in the OS keychain if none is configured. Returns a status string.

    Never prints the key. Interactive callers get a no-echo prompt; others get an actionable hint.
    """
    if provider_key_present(runtime_keys):
        return "present"
    if not interactive:
        return "not-configured (run `codna key set <provider>`)"
    try:
        answer = input(
            "\nNo local LLM provider key found. Add one now so `codna fix` runs on-device?\n"
            "  Enter a provider (anthropic/openai/gemini/…) or press Enter to skip: "
        ).strip()
    except EOFError:
        return "not-configured (run `codna key set <provider>`)"
    if not answer:
        return "skipped (run `codna key set <provider>` later)"
    try:
        import getpass

        secret = getpass.getpass(f"  Paste the {answer} key (input hidden): ").strip()
        if not secret:
            return "skipped"
        keystore.set_key(answer, secret)   # return (the env-var name) intentionally unused
    except keystore.KeystoreError as exc:
        print(f"  could not store the key: {exc}", file=sys.stderr)
        return "not-configured"
    # Literal status only — never interpolate anything on the set_key(secret) dataflow into a returned/
    # printed string (CodeQL: clear-text logging of sensitive info; the value is the key NAME, not the
    # secret, but keep the status provably credential-free).
    return "stored"


def cmd_key(args) -> int:
    """Manage the on-device LLM provider key(s) used by local `codna fix` (OS keychain; local BYOK)."""
    action = getattr(args, "key_action", None)
    try:
        if action == "list":
            print(json.dumps(
                {"ok": True, "keys": keystore.stored_key_names(), "keychain": keystore.available()},
                indent=2,
            ))
            return 0
        if action == "rm":
            existed = keystore.delete_key(args.provider)
            print(json.dumps({"ok": True, "removed": existed, "provider": args.provider}, indent=2))
            return 0
        if action == "set":
            # Read the secret WITHOUT echo / without argv (no shell-history leak). --stdin for CI.
            if getattr(args, "stdin", False):
                secret = sys.stdin.readline().strip()
            else:
                import getpass

                secret = getpass.getpass(f"Paste the {args.provider} key (input hidden): ").strip()
            stored = keystore.set_key(args.provider, secret)
            print(json.dumps({"ok": True, "stored": stored}, indent=2))  # value never printed
            return 0
    except keystore.KeystoreError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": False, "error": "usage: codna key {set|list|rm} …"}, indent=2), file=sys.stderr)
    return 2


def register(sub) -> None:
    """Attach `codna key {set,list,rm}` to the top-level subparsers.

    Lives here rather than in `cli.py` for the reason `impact_cli.register_cli` does: a command's
    parser belongs with its handler, and cli.py is held under a hard line ceiling by
    `test_modularity` — which is what forced this move when `codna ci` was added.
    """
    pk = sub.add_parser("key", help="Manage the on-device LLM provider key for local `codna fix` (OS keychain).")
    pk_sub = pk.add_subparsers(dest="key_action")
    pk_set = pk_sub.add_parser("set", help="Store a provider key (read via hidden prompt; never on the command line).")
    pk_set.add_argument("provider", help="provider name (anthropic/openai/gemini/…) or *_API_KEY variable name")
    pk_set.add_argument("--stdin", action="store_true", help="read the key from stdin (for CI) instead of prompting")
    pk_set.set_defaults(func=cmd_key)
    pk_list = pk_sub.add_parser("list", help="List which provider keys are stored (values never shown).")
    pk_list.set_defaults(func=cmd_key)
    pk_rm = pk_sub.add_parser("rm", help="Remove a stored provider key.")
    pk_rm.add_argument("provider", help="provider name or *_API_KEY variable name")
    pk_rm.set_defaults(func=cmd_key)
