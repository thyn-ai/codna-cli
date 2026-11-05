"""Runtime-readiness probe for the Telys-backed suites, with three honest outcomes.

WHAT THIS USED TO GET WRONG. The probe asked one question — "does a kernel dylib AND a bare ``import telys``
work?" — and answered it by importing ``telys`` IN THE TEST PROCESS AT COLLECTION TIME. Both halves were
wrong, and together they are how ``test_mcp_recall.py::test_recall_happy_path`` rotted from green to red
without anyone changing the test:

* **It cached the wrong SDK.** Any importable ``telys`` satisfied the probe, so an old system-wide
  ``telys`` (0.1.0b3, which predates ``TEXT_COLUMN``) passed. That module then stayed in ``sys.modules`` for
  the whole session, so the product's own later ``from telys import ...`` was served the stale copy — no
  amount of correct ``sys.path`` wiring downstream could undo it. The probe must not decide which SDK the
  product binds, so it now restores ``sys.modules`` exactly as it restores ``sys.path`` and the env.
* **"No kernel" was treated as "skip".** A resolved install whose packages cannot import is a BROKEN
  install, not an absent one; skipping it is what kept the failure invisible.

The state is now one of:
    ready   the product's own Telys surface imports and an engine constructs here
    absent  nothing claims to provide Telys here — a skip is honest, and CI asserts this does not happen
    broken  something claims to provide it and cannot be used — never a skip

``CODNA_REQUIRE_TELYS_RUNTIME=1`` turns ``absent`` into a failure, so a CI gate can prove it did not skip.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

KERNEL_ENV_NAMES = ("TELYS_KERNEL", "AME_KERNEL", "CODNA_TELYS_KERNEL")
# CI sets this so a missing install is a RED STEP, not a green job that skipped the tests it exists for.
REQUIRE_ENV = "CODNA_REQUIRE_TELYS_RUNTIME"

READY = "ready"
ABSENT = "absent"
BROKEN = "broken"


@dataclass(frozen=True)
class TelysRuntimeStatus:
    """The probe's verdict: ``kind`` plus the detail a skip/failure message must carry."""

    kind: str
    detail: str
    version: str | None = None

    @property
    def ready(self) -> bool:
        return self.kind == READY

    def skip_reason(self) -> str:
        return f"Telys runtime {self.kind}: {self.detail}"


def _modules_of_interest() -> dict[str, object]:
    """The modules the probe may import and must therefore restore: the SDK and the engine it loads."""
    return {
        name: module
        for name, module in list(sys.modules.items())
        if name == "telys" or name.startswith("telys.") or name == "memengine" or name.startswith("memengine.")
    }


def _restore_modules(saved: dict[str, object]) -> None:
    """Undo every module the probe imported, so the session binds whatever the PRODUCT resolves.

    Without this the probe's import wins the session: a stale ``telys`` cached during collection is what
    later served the product, which is the defect this file now guards. Modules that were already imported
    before the probe are left untouched — the probe must not unload someone else's import.
    """
    for name in list(sys.modules):
        if (name == "telys" or name.startswith("telys.") or name == "memengine"
                or name.startswith("memengine.")) and name not in saved:
            del sys.modules[name]
    sys.modules.update(saved)


def _probe() -> TelysRuntimeStatus:
    """Resolve and exercise the surface the memory tool needs.

    On success the SDK binding and ``sys.path`` activation are LEFT IN PLACE: they are the product's own
    wiring, and a module that imports ``telys`` later in the same session (``test_memory_wrapper.py`` does,
    at module scope) must bind the same install the product will use. Restoring them would re-open exactly
    the hole this file now guards — the session importing an unrelated system-wide ``telys``.

    Only the ENV is always restored. ``_configure_telys_kernel_env`` writes ``TELYS_KERNEL``, and
    ``memengine.mojo_backend`` latches it once at import; a collection-time probe must not decide that for
    every later test. The product calls the same resolver again when it actually needs it.
    """
    saved_path = list(sys.path)
    saved_env = {name: os.environ.get(name) for name in KERNEL_ENV_NAMES}
    saved_modules = _modules_of_interest()
    keep_binding = False
    # Bound BEFORE the try, and not left to the `from codna import memory` inside it: the `except` handler
    # needs a package path to report on, and if THAT import is what failed, a name bound inside the `try`
    # would be unbound in the handler and the handler itself would raise NameError — breaking the
    # "nothing here may raise" contract on exactly the machine that most needs a verdict.
    package_file = str(Path(__file__).resolve().parent.parent / "codna" / "memory.py")
    try:
        from codna import memory as memory_module
        from codna import memory_install

        package_file = memory_module.__file__

        # The product's own activation first: the probe must judge the SAME install the tool will use,
        # not a hand-configured environment. (This is what the manual-PYTHONPATH runs were proving.)
        memory_module._activate_install_site_packages(package_file)  # noqa: SLF001 — the probe exists to exercise it
        install_roots = memory_install.candidate_install_site_dirs(package_file)
        memory_module._configure_telys_kernel_env()  # noqa: SLF001
        claims_install = bool(install_roots)

        # Bind FRESH from the activated path. A `telys` cached earlier in the session (by this file on a
        # previous run, or by an unrelated module) is precisely the stale-binding failure mode, so it must
        # not satisfy the probe. `invalidate_caches()` because CPython caches per-directory finder results,
        # and a miss recorded before the path entry existed would otherwise be reused (see
        # `cli/tests/test_pynative_kernel_absent.py`, which pairs the two the same way).
        _restore_modules({})
        importlib.invalidate_caches()
        try:
            import telys
            from telys import Telys
            from telys.embedding import AlgentaMultigramEmbedder
            from telys.runtime import TEXT_COLUMN
        except ImportError as exc:
            where = ", ".join(f"{source}:{path}" for source, path in install_roots) or "no Codna install root"
            return TelysRuntimeStatus(
                ABSENT if not claims_install else BROKEN,
                f"the Telys SDK is not importable ({exc}); resolved installs: {where}",
            )

        version = getattr(telys, "__version__", None)
        # The symbol contract the product itself needs: `cli/memengine/native_runtime.py` imports
        # TEXT_COLUMN at module scope, so a 0.1.0b3-era SDK satisfies `import telys` and still cannot be
        # used. Declared in `cli/pyproject.toml`; asserted here so a floor regression fails loudly.
        AlgentaMultigramEmbedder()
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            Telys(os.path.join(scratch, "engine"), embedding_providers={})
        keep_binding = True
        return TelysRuntimeStatus(
            READY,
            f"telys {version or 'unknown'} imported from {telys.__file__} and an engine constructed"
            + (f" (TEXT_COLUMN={TEXT_COLUMN!r})" if TEXT_COLUMN else ""),
            version=version,
        )
    except Exception as exc:  # noqa: BLE001 — the verdict IS the outcome; nothing here may raise
        # Distinguish "nothing was ever installed here" from "an install exists and is unusable". The
        # no-kernel error is raised by telys itself when neither a kernel nor a verified runtime is
        # present — that is genuine ABSENCE, and calling it BROKEN would turn a clean local skip into a
        # hard error on every machine that simply has not installed Telys.
        #
        # Every step here is guarded: this handler MUST NOT raise, and it may be reached because
        # `from codna import memory` itself failed (hence `package_file` being bound before the try).
        try:
            from codna import memory as _memory
            from codna import memory_install as _memory_install

            missing_kernel = "on-device kernel" in str(exc) and "not installed" in str(exc)
            claims = bool(_memory_install.candidate_install_site_dirs(package_file))
            claims = claims or bool(_memory._resolve_telys_kernel(configure_env=False).get("found"))  # noqa: SLF001
        except Exception:  # noqa: BLE001 — classification is best-effort; BROKEN is the safe verdict
            missing_kernel, claims = False, True
        kind = ABSENT if (missing_kernel and not claims) else BROKEN
        return TelysRuntimeStatus(kind, f"{type(exc).__name__}: {exc}")
    finally:
        # `sys.path` and the module binding survive a READY verdict (see the docstring); a failed probe
        # leaves no trace at all.
        if not keep_binding:
            sys.path[:] = saved_path
            _restore_modules(saved_modules)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def telys_runtime_status() -> TelysRuntimeStatus:
    """The full verdict (``ready`` / ``absent`` / ``broken``) for the current checkout."""
    return _probe()


def telys_kernel_ready() -> bool:
    """True when the Telys surface is usable here — evaluated WITHOUT leaking into the session.

    Kept under its historical name because ``test_memory_wrapper.py`` gates its module on it; the meaning
    is now "the product's Telys surface imports and an engine constructs", which is what those tests need.
    A kernel dylib is no longer required: the in-process LocalRuntime serves recall without one, and
    demanding a dylib turned a working install into a skip.
    """
    return telys_runtime_status().ready


def require_telys_runtime() -> TelysRuntimeStatus:
    """The verdict, raising instead of returning a skippable state, for use at module scope in a test.

    ``broken`` always raises — a present-but-unusable install must never be reported as a skip.
    ``absent`` raises only when ``CODNA_REQUIRE_TELYS_RUNTIME=1`` (CI), and names the variable so a local
    run on a machine without Telys stays a clean skip.
    """
    status = telys_runtime_status()
    if status.kind == BROKEN:
        raise AssertionError(
            f"A Telys install is present but unusable: {status.detail}. This is not a skip — the "
            "install exists and the product cannot import it, so the gated tests would hide a real "
            "defect. Fix the install (or remove it) instead of widening the skip."
        )
    if status.kind == ABSENT and os.environ.get(REQUIRE_ENV) == "1":
        raise AssertionError(
            f"No usable Telys runtime, and {REQUIRE_ENV}=1 requires one: {status.detail}. The CI gate "
            "sets that variable so this job cannot pass by skipping the tests it exists to run."
        )
    return status
