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
import re
import types
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DURABLE_EXTRA",
    "DURABLE_PINS",
    "DURABLE_RANGES",
    "INSTALL_HINT",
    "PLUGIN_DISTRIBUTION",
    "PLUGIN_PIN",
    "PLUGIN_RANGE",
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

    def accepts(self, version: tuple[tuple[int, ...], bool]) -> bool:
        release, prerelease = version
        width = max(len(release), len(self.minimum), len(self.maximum))
        release = release + (0,) * (width - len(release))
        minimum = self.minimum + (0,) * (width - len(self.minimum))
        maximum = self.maximum + (0,) * (width - len(self.maximum))
        if prerelease:
            return minimum <= release < maximum
        return minimum <= release < maximum


# The checkpoint package is pinned as well as declared in the extra because it carries the
# checkpoint semantics `interrupt()`/`Command(resume=...)` depend on. All entries are kept in sync
# with cli/constraints-durable.txt by tests/durable/test_durable_extra.py, so a release cannot pin
# one version and ship another.
DURABLE_PINS: Final[tuple[Requirement, ...]] = (
    Requirement(
        distribution="temporalio",
        module="temporalio",
        pin="1.35.0",
        minimum=(1, 34),
        maximum=(2,),
    ),
    Requirement(
        distribution="langgraph",
        module="langgraph",
        pin="1.2.20",
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
        pin="4.3.1",
        minimum=(4,),
        maximum=(5,),
    ),
)

# The plugin itself. It IS the Public-Preview surface — the code that decides how a graph node
# becomes a Temporal activity, and therefore the workflow's history semantics.
#
# Up to ``temporalio`` 1.34 the plugin was code INSIDE the ``temporalio`` distribution, so pinning
# ``temporalio`` pinned it and there was nothing else to check. From 1.35 it is a separately
# versioned distribution, ``temporalio-langgraph``, so the ``temporalio`` pin no longer constrains
# it. ``temporalio.contrib.langgraph`` still imports — it forwards to ``temporalio.langgraph``.
PLUGIN_MODULE: Final = "temporalio.contrib.langgraph"
PLUGIN_DISTRIBUTION: Final = "temporalio-langgraph"

# The plugin's pin and window, kept OUT of ``DURABLE_PINS`` on purpose. That tuple is the version
# GATE, and its ``temporalio>=1.34`` window admits 1.34, where ``temporalio-langgraph`` does not
# exist (the plugin was vendored) — requiring the distribution there would refuse a combination the
# extra still declares supported. So the plugin is checked separately, and only WHEN PRESENT: on a
# 1.34 install it is absent and the ``temporalio`` pin above is what guards the plugin. Where it IS
# present it is the highest-risk surface in the stack, so it is range-checked like the rest.
#
# ``>=0.1,<0.2``: a 0.x minor is allowed to break, so moving off 0.1 must be a deliberate, tested
# bump rather than something a resolver can do unnoticed.
PLUGIN_PIN: Final = "0.1.0"
PLUGIN_RANGE: Final = ">=0.1,<0.2"
PLUGIN_MINIMUM: Final[tuple[int, ...]] = (0, 1)
PLUGIN_MAXIMUM: Final[tuple[int, ...]] = (0, 2)


def _plugin_accepts(version: tuple[tuple[int, ...], bool]) -> bool:
    """Whether an installed ``temporalio-langgraph`` is inside the supported window.

    Same comparison as :meth:`Requirement.accepts`, written out here because the plugin has no
    ``Requirement`` of its own — see ``PLUGIN_DISTRIBUTION`` for why it is not in ``DURABLE_PINS``.

    The prerelease flag is unpacked but deliberately not special-cased: a prerelease is in-window
    iff its release component is, so ``0.1.0b1`` is accepted and ``0.2.0a1`` is not (asserted by
    the window matrix). Unpacking keeps this structurally identical to ``Requirement.accepts`` and
    makes a malformed version tuple a ``ValueError`` here rather than a silently wrong comparison.
    """
    release, _prerelease = version
    width = max(len(release), len(PLUGIN_MINIMUM), len(PLUGIN_MAXIMUM))
    padded = release + (0,) * (width - len(release))
    minimum = PLUGIN_MINIMUM + (0,) * (width - len(PLUGIN_MINIMUM))
    maximum = PLUGIN_MAXIMUM + (0,) * (width - len(PLUGIN_MAXIMUM))
    return minimum <= padded < maximum

# Human-readable ranges, mirroring the pyproject extra. Asserted equal to that extra by the tests.
DURABLE_RANGES: Final[dict[str, str]] = {
    "temporalio": ">=1.34,<2",
    "langgraph": ">=1,<2",
    "langgraph-checkpoint": ">=4,<5",
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


def _parse_version(raw: str) -> tuple[tuple[int, ...], bool]:
    """Return the numeric release and whether it is a prerelease, without a packaging dependency."""
    base = raw.split("+", 1)[0]
    release: list[int] = []
    for part in base.split("-", 1)[0].split("."):
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
    prerelease = re.search(r"(?i)(dev|a|alpha|b|beta|rc|c|pre|preview)[.-]?\d*$", base) is not None
    return tuple(release), prerelease


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

    # The LangGraph plugin's own version. Since `temporalio` 1.35 the plugin is a SEPARATE
    # distribution (`temporalio-langgraph`) with its own release cadence, and it owns the
    # node→activity mapping — i.e. the history semantics. It is deliberately not in `DURABLE_PINS`
    # (see `PLUGIN_DISTRIBUTION`), so it would otherwise go unchecked at runtime, which would make
    # the pin a packaging detail rather than a guard.
    plugin_version = _installed_version(PLUGIN_DISTRIBUTION)
    # `None` (not importable as a distribution) and an empty string both mean ABSENT, which is legal
    # — see `PLUGIN_DISTRIBUTION` for why a 1.34 install has no plugin distribution to report.
    if plugin_version:
        try:
            plugin_parsed = _parse_version(plugin_version)
        except ValueError as exc:
            raise DurableUnavailableError(
                f"cannot interpret the installed {PLUGIN_DISTRIBUTION} version {plugin_version!r}.",
                code="durable_version_unsupported",
                details={"distribution": PLUGIN_DISTRIBUTION, "installed": plugin_version},
            ) from exc
        if not _plugin_accepts(plugin_parsed):
            raise DurableUnavailableError(
                f"{PLUGIN_DISTRIBUTION} {plugin_version} is outside the supported window "
                f"(pinned {PLUGIN_PIN}).",
                code="durable_version_unsupported",
                details={
                    "distribution": PLUGIN_DISTRIBUTION,
                    "installed": plugin_version,
                    "pinned": PLUGIN_PIN,
                    "range": PLUGIN_RANGE,
                },
                hint=f"pip install {PLUGIN_DISTRIBUTION}=={PLUGIN_PIN}",
            )
        versions[PLUGIN_DISTRIBUTION] = plugin_version

    return DurableStack(
        temporalio=importlib.import_module("temporalio"),
        langgraph=importlib.import_module("langgraph"),
        langchain_core=importlib.import_module("langchain_core"),
        langgraph_plugin=plugin,
        versions=versions,
    )
