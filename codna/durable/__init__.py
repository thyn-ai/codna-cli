"""Durable execution for Codna runs — the ``codna[durable]`` extra.

A Codna run is a long-lived task (plan → implement → verify → review → open PR → wait for a
human) that must survive a laptop sleeping, a sidecar ``kill -9`` and a worker restart without
re-running a paid turn. This package is the substrate for that: Temporal owns the durable
execution, LangGraph owns the task graph, and an append-only hash-chained ledger owns the audit
trail.

``temporalio``/``langgraph``/``langchain-core`` are an OPTIONAL extra (``codna[durable]``). They
are deliberately NOT base dependencies: they add ~40 MB and a workflow sandbox that a plain
``pip install codna`` never needs. Every entry point therefore calls :func:`require_durable`
first and gets a structured, actionable failure instead of a bare ``ModuleNotFoundError``.

Importing this module never imports ``temporalio`` — the base suite (and ``codna --version``)
must stay green without the extra. Only :func:`require_durable` imports it, and only when a
durable command actually runs.
"""
from __future__ import annotations

import importlib
import importlib.util
import types
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DURABLE_EXTRA",
    "DURABLE_PINS",
    "DURABLE_RANGES",
    "INSTALL_HINT",
    "DurableStack",
    "DurableUnavailableError",
    "Requirement",
    "require_durable",
]

DURABLE_EXTRA: Final = "durable"
INSTALL_HINT: Final = "pip install 'codna[durable]'"


@dataclass(frozen=True)
class Requirement:
    """One distribution the durable stack needs: its import name, the recorded pin and a range.

    ``pin`` is the exact version recorded in ``cli/constraints-durable.txt`` (what the release
    CI installs); ``minimum``/``maximum`` is the compatible window the extra declares. A version
    outside the window fails fast, because the LangGraph plugin is Public Preview: an
    unannounced minor bump can change workflow history semantics, which is exactly the class of
    change that breaks replay of runs already in flight.
    """

    distribution: str
    module: str
    pin: str
    minimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def accepts(self, version: tuple[int, ...]) -> bool:
        return self.minimum <= version < self.maximum


# The two LangGraph entries are transitive, pinned because they carry the checkpoint semantics
# `interrupt()`/`Command(resume=...)` depend on. Kept in sync with cli/constraints-durable.txt by
# tests/durable/test_durable_extra.py, so a release cannot pin one version and ship another.
DURABLE_PINS: Final[tuple[Requirement, ...]] = (
    Requirement(
        distribution="temporalio",
        module="temporalio",
        pin="1.34.0",
        minimum=(1, 34),
        maximum=(2,),
    ),
    Requirement(
        distribution="langgraph",
        module="langgraph",
        pin="1.2.17",
        minimum=(1,),
        maximum=(2,),
    ),
    Requirement(
        distribution="langchain-core",
        module="langchain_core",
        pin="1.7.0",
        minimum=(1,),
        maximum=(2,),
    ),
    Requirement(
        distribution="langgraph-checkpoint",
        module="langgraph.checkpoint",
        pin="4.3.0",
        minimum=(4,),
        maximum=(5,),
    ),
)

# The plugin itself. It lives inside temporalio (``temporalio[langgraph]``), so it has no separate
# distribution to pin — but it IS the Public-Preview surface, hence the ``temporalio`` pin above
# and the explicit import check here.
PLUGIN_MODULE: Final = "temporalio.contrib.langgraph"

# Human-readable ranges, mirroring the pyproject extra. Asserted equal to that extra by the tests.
DURABLE_RANGES: Final[dict[str, str]] = {
    "temporalio": ">=1.34,<2",
    "langgraph": ">=1,<2",
    "langchain-core": ">=1,<2",
}


class DurableUnavailableError(RuntimeError):
    """Durable support cannot be used. Always carries a machine-readable ``code`` and a hint."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, object] | None = None,
        hint: str = INSTALL_HINT,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})
        self.hint = hint

    def as_dict(self) -> dict[str, object]:
        """Structured error body (the shape the CLI prints and the Run API returns)."""
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "details": self.details,
                "hint": self.hint,
            }
        }

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.args[0]} (code={self.code})"


@dataclass(frozen=True)
class DurableStack:
    """The imported durable stack, so callers import each module exactly once.

    A structured object rather than a bare ``import`` because every durable entry point needs
    the same four modules and none of them may be imported at module scope by this package.
    """

    temporalio: types.ModuleType
    langgraph: types.ModuleType
    langchain_core: types.ModuleType
    langgraph_plugin: types.ModuleType
    versions: dict[str, str]


def _parse_version(raw: str) -> tuple[int, ...]:
    """Parse the leading numeric release of a version string (``1.34.0b2`` → ``(1, 34, 0)``).

    Deliberately tiny: this must not add a ``packaging`` dependency to a base install. Only the
    release segment matters for the compatibility window; a pre-release sorts inside its range.
    """
    release: list[int] = []
    for part in raw.split("+", 1)[0].split("-", 1)[0].split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        release.append(int(digits))
    if not release:
        raise ValueError(f"unparseable version {raw!r}")
    return tuple(release)


def _installed_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python >= 3.12 always has it
        return None
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _missing_modules(requirements: tuple[Requirement, ...]) -> list[str]:
    missing: list[str] = []
    for requirement in requirements:
        try:
            if importlib.util.find_spec(requirement.module) is None:
                missing.append(requirement.distribution)
        except (ImportError, ValueError):  # a broken/partial install: treat as missing
            missing.append(requirement.distribution)
    return missing


def require_durable() -> DurableStack:
    """Import and validate the durable stack, or fail fast with a structured, actionable error.

    Never falls back silently: a durable command that cannot guarantee durable execution must not
    pretend to run one. The three failure modes are distinct codes so a caller (or the Run API)
    can tell a missing extra from an unsupported version:

    * ``durable_extra_missing`` — the extra was never installed;
    * ``durable_plugin_missing`` — ``temporalio`` is present without its LangGraph contrib;
    * ``durable_version_unsupported`` — installed, but outside the pinned compatibility window.
    """
    missing = _missing_modules(DURABLE_PINS)
    if missing:
        raise DurableUnavailableError(
            "Durable runs need the `codna[durable]` extra "
            f"(missing: {', '.join(missing)}). Install it with: {INSTALL_HINT}",
            code="durable_extra_missing",
            details={"missing": missing, "install": INSTALL_HINT},
        )

    try:
        plugin = importlib.import_module(PLUGIN_MODULE)
    except ImportError as exc:
        raise DurableUnavailableError(
            "temporalio is installed without its LangGraph contrib. Install it with: "
            "pip install 'temporalio[langgraph]'",
            code="durable_plugin_missing",
            details={"module": PLUGIN_MODULE, "install": "pip install 'temporalio[langgraph]'"},
        ) from exc

    versions: dict[str, str] = {}
    for requirement in DURABLE_PINS:
        installed = _installed_version(requirement.distribution)
        if installed is None:
            raise DurableUnavailableError(
                f"{requirement.distribution} is importable but not installed as a distribution.",
                code="durable_extra_missing",
                details={"distribution": requirement.distribution, "install": INSTALL_HINT},
            )
        try:
            parsed = _parse_version(installed)
        except ValueError as exc:
            raise DurableUnavailableError(
                f"cannot interpret the installed {requirement.distribution} version {installed!r}.",
                code="durable_version_unsupported",
                details={"distribution": requirement.distribution, "installed": installed},
            ) from exc
        if not requirement.accepts(parsed):
            raise DurableUnavailableError(
                f"{requirement.distribution} {installed} is outside the supported window "
                f"(pinned {requirement.pin}).",
                code="durable_version_unsupported",
                details={
                    "distribution": requirement.distribution,
                    "installed": installed,
                    "pinned": requirement.pin,
                    "range": DURABLE_RANGES.get(requirement.distribution, ""),
                },
                hint=f"pip install {requirement.distribution}=={requirement.pin}",
            )
        versions[requirement.distribution] = installed

    return DurableStack(
        temporalio=importlib.import_module("temporalio"),
        langgraph=importlib.import_module("langgraph"),
        langchain_core=importlib.import_module("langchain_core"),
        langgraph_plugin=plugin,
        versions=versions,
    )
