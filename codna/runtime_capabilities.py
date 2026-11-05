"""Runtime capability decisions — one source of truth for "can this runtime do X?".

`hasattr(collection, "export_compact")` cannot answer any of this: every method on the SDK's
`RuntimeHandle` contract EXISTS on every runtime, so the attribute is always present even when calling it
raises `NotImplementedError`. That is why a missing capability used to surface as a crash inside the
caller instead of a clean, early refusal.

The decision is made here, once, and used by both the product and the tests, so they cannot drift apart:

  1. a runtime that declares the flag answers definitively;
  2. otherwise (an SDK older than the flag) a method inherited unchanged from `RuntimeHandle` is the base
     class's ``raise NotImplementedError`` body — present, supporting nothing;
  3. anything else is assumed capable, which keeps pre-flag runtimes that DID implement the method working.

This module lives in `codna` deliberately. A verified `telys runtime install` ships a SLIM `memengine`
(`native_runtime.py` only), so capability logic placed in `memengine` would be absent from exactly the
production installs that need it.
"""

from __future__ import annotations

COMPACT_EXPORT = "supports_compact_export"
COMPACT_OPEN = "supports_compact_open"


def _overrides(runtime, method: str) -> bool:
    """True when `runtime` supplies its own `method` rather than the SDK's unimplemented base body.

    Resolved on the INSTANCE (not the class): a runtime may provide the method per-instance, and a bound
    method is a fresh object on every access, so compare the underlying function. With no SDK importable
    there is no base class to compare against, so assume it does (the caller's error handling still
    applies) rather than refusing a runtime that may well work.
    """
    impl = getattr(runtime, method, None)
    if impl is None:
        return False
    func = getattr(impl, "__func__", impl)
    try:
        from telys.runtime import RuntimeHandle
    except Exception:  # noqa: BLE001 — no SDK base class to compare against
        return True
    return func is not getattr(RuntimeHandle, method, object())


def _declares(runtime, capability: str, method: str) -> tuple[bool, str]:
    name = type(runtime).__name__
    flag = getattr(runtime, capability, None)
    if flag is not None:
        return bool(flag), f"{name}.{capability}={flag}"
    if not _overrides(runtime, method):
        return False, f"{name} inherits the SDK's unimplemented {method}"
    return True, f"{name} implements {method} (no {capability} flag)"


def supports_compact_export(runtime) -> tuple[bool, str]:
    """(supported, detail) for WRITING a compact serve artifact with `runtime`."""
    return _declares(runtime, COMPACT_EXPORT, "export_compact")


def require_compact_export(runtime, error_type):
    """Raise `error_type` unless `runtime` can write a compact serve artifact.

    Kept here rather than at the call site so `memory.py` does not grow past its 1000-line ceiling
    (`test_modularity.py`), and because this is the same decision `supports_compact_export` makes —
    one source of truth, two presentations: a predicate for tests, an exception for the product.
    """
    supported, detail = supports_compact_export(runtime)
    if not supported:
        raise error_type(
            f"the resolved runtime cannot export compact serve artifacts ({detail}) — "
            "export from the local runtime (TELYS_RUNTIME=local)"
        )


def supports_compact_open(runtime) -> tuple[bool, str]:
    """(supported, detail) for READING a compact serve artifact with `runtime`."""
    return _declares(runtime, COMPACT_OPEN, "open_compact")
