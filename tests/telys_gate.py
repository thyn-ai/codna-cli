"""Kernel-readiness probes that do not mutate the whole pytest session.

``codna.memory._configure_telys_kernel_env()`` resolves the on-device kernel and, on success, writes the
resolved path into ``os.environ["TELYS_KERNEL"]`` — which is the right thing for a CLI invocation and the wrong
thing for collection. pytest imports every test module during collection, so a module-level probe leaves that
override in place for the ENTIRE session: every later test then resolves a kernel it never asked for, and
``memengine.mojo_backend`` (which reads the variable once, at import) can latch onto a path this module never
meant to advertise.

The probe therefore snapshots what it touches and restores it unconditionally, whether the kernel turned out to
be available or not. ``sys.path`` is snapshotted too: importing ``telys`` prepends the verified runtime's
``pysite``, and a module-level probe should not decide what that means for everyone else.
"""
from __future__ import annotations

import os
import sys

KERNEL_ENV_NAMES = ("TELYS_KERNEL", "AME_KERNEL", "CODNA_TELYS_KERNEL")


def telys_kernel_ready() -> bool:
    """True when the Telys kernel loads here — evaluated WITHOUT leaking the resolution into the session.

    Call this at module scope in place of ``_configure_telys_kernel_env()`` + a bare import: the resolution it
    performs is discarded, so the module is free to skip (or not) without changing any other file's environment.
    """
    saved_path = list(sys.path)
    saved_env = {name: os.environ.get(name) for name in KERNEL_ENV_NAMES}
    try:
        from codna.memory import _configure_telys_kernel_env

        _configure_telys_kernel_env()
        from telys import scope_key  # noqa: F401
        from telys.embedding import AlgentaBigramEmbedder

        AlgentaBigramEmbedder()  # loads the kernel dylib; raises if absent
        return True
    except Exception:  # noqa: BLE001 - any failure means "not available here", which the caller turns into a skip
        return False
    finally:
        sys.path[:] = saved_path
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
