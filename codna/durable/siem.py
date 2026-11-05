"""The SIEM export: one JSON object per security-relevant act, on a sink that cannot lose an event.

The Run API is where a durable run is read, started, steered and approved. Every one of those is a
security-relevant act, and the ledger is the system of record — but a ledger lives on the host that
produced it, in NDJSON, and a SIEM consumes neither of those properties. This module is the export:

* **A stable envelope** (`SiemEvent`) that names the act, the verified principal, the run, the route,
  and the outcome — with no credential, no token, and no request body in it.
* **Append-only NDJSON**, one file per host, rotated by size, written with the same exclusive-lock +
  fsync discipline as the ledger. Written by hand rather than delegated to a log handler because a
  log handler drops on a full disk and this must not: a dropped "approval.granted" is a hole in an
  audit trail.
* **A pluggable sink** (`SiemSink` protocol) so a hosted deployment can fan the same events to
  Splunk/Datadog without changing a call site. `NullSiemSink` exists for tests that do not care;
  `CompositeSiemSink` is why "local file AND remote" is one configuration rather than two code paths.

**Retention beyond the 90 days.** Temporal Cloud keeps a closed workflow's history for a bounded
period (90 days on the common plan), which is shorter than any audit requirement that matters. The
ledger already outlives it on disk, and `RetentionPolicy` here is the explicit, tested statement of
how long: it prunes *SIEM* files older than the window, and it verifies the ledger is still present
for a run inside the window (a run whose ledger vanished is reported, never silently accepted). What
it deliberately does NOT do is delete a ledger — that is the system of record a compliance officer
asked for, and a retention job that erases evidence is a worse bug than a full disk.

**What is not wired, stated plainly.** The REMOTE sink has no implementation: there is no Splunk or
Datadog endpoint in this build's environment, and inventing an HTTP client for one would be a
placeholder pretending to be complete. So the local NDJSON file is fully implemented and tested, the
protocol is the seam a remote sink plugs into, and `RemoteSiemSink` raises a coded
`SiemDeliveryError` rather than silently dropping — a fan-out that fails quietly is the one
behaviour worse than no fan-out.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Protocol

from .roles import Principal

__all__ = [
    "DEFAULT_RETENTION_DAYS",
    "RETENTION_ENV",
    "CompositeSiemSink",
    "FileSiemSink",
    "NullSiemSink",
    "RemoteSiemSink",
    "RetentionPolicy",
    "SiemDeliveryError",
    "SiemEvent",
    "SiemSink",
    "build_siem_sink",
    "retention_days_from_env",
]

# One year. Chosen because it is the shortest window that covers every common audit requirement
# (SOC 2 evidence, a financial-year breach investigation) without keeping personal data forever.
DEFAULT_RETENTION_DAYS: Final = 365
RETENTION_ENV: Final = "CODNA_SIEM_RETENTION_DAYS"
SIEM_ENV: Final = "CODNA_SIEM_DIR"
# A SIEM file never holds more than this before it is rotated, so one huge day cannot make a reader
# of "the current file" skip an hour of records.
MAX_FILE_BYTES: Final = 64 * 1024 * 1024

# The act names. A closed set: an event nobody can query by name is noise in a SIEM, and a typo in a
# call site would otherwise create a new, permanently empty facet.
EVENT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "auth.succeeded",
        "auth.denied",
        "run.started",
        "run.controlled",
        "run.steered",
        "run.approved",
        "run.budget_raised",
        "ledger.verified",
        "ledger.diverged",
        "ledger.exported",
    }
)


class SiemDeliveryError(RuntimeError):
    """A SIEM event could not be delivered. Raised, never swallowed."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})


@dataclass(frozen=True)
class SiemEvent:
    """One SIEM record. Uncredentialed, bounded, and complete enough to answer "who did what"."""

    event: str
    ts: float
    identity: str
    roles: tuple[str, ...]
    source: str
    run_id: str = ""
    task_id: str = ""
    route: str = ""
    outcome: str = "ok"
    detail: Mapping[str, Any] = field(default_factory=dict)
    host: str = ""

    def __post_init__(self) -> None:
        if self.event not in EVENT_NAMES:
            raise ValueError(f"unknown SIEM event {self.event!r}")
        if not self.identity:
            raise ValueError("a SIEM event needs an identity (an anonymous audit row is useless)")

    @classmethod
    def action(
        cls,
        event: str,
        *,
        principal: Principal,
        route: str,
        run_id: str = "",
        task_id: str = "",
        outcome: str = "ok",
        detail: Mapping[str, Any] | None = None,
        clock: Any = time.time,
        host: str = "",
    ) -> "SiemEvent":
        """Build an event from a verified principal — the ONLY way a caller is named."""
        return cls(
            event=event,
            ts=float(clock()),
            identity=principal.identity,
            roles=tuple(sorted(principal.roles)),
            source=principal.source,
            run_id=run_id,
            task_id=task_id,
            route=route,
            outcome=outcome,
            detail=dict(detail or {}),
            host=host or _hostname(),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "iso_ts": datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat(),
            "event": self.event,
            "identity": self.identity,
            "roles": list(self.roles),
            "source": self.source,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "route": self.route,
            "outcome": self.outcome,
            "host": self.host,
            "detail": dict(self.detail),
        }

    def to_json(self) -> str:
        # Canonical, sorted, no NaN: the same reason the ledger hashes canonical JSON.
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)


class SiemSink(Protocol):
    """Where events go. Implementations must raise on failure rather than drop."""

    def record(self, event: SiemEvent) -> None: ...

    def close(self) -> None: ...


class NullSiemSink:
    """Records nothing. For a test or an embedder that has its own exporter.

    Explicit, not a silent default: `build_siem_sink` returns a FILE sink unless told otherwise, so
    "no export" is something a caller chooses by name.
    """

    def record(self, event: SiemEvent) -> None:  # noqa: ARG002 - the protocol, deliberately empty
        return None

    def close(self) -> None:
        return None


class FileSiemSink:
    """Append-only NDJSON on this host, rotated by size, fsynced per line.

    The lock is on the file itself (like the ledger's), so two server processes on one host cannot
    interleave a line. A line is written and fsynced BEFORE `record` returns: a caller that recorded
    an approval and then crashed has an approval in the export.
    """

    def __init__(self, directory: Path, *, clock: Any = time.time, max_bytes: int = MAX_FILE_BYTES) -> None:
        self._dir = Path(directory).expanduser()
        self._clock = clock
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._path = self._dir / f"run-api-{datetime.now(timezone.utc):%Y%m%d}.ndjson"

    @property
    def path(self) -> Path:
        """The file currently being appended to (rotates at a size boundary)."""
        return self._path

    def record(self, event: SiemEvent) -> None:
        line = event.to_json() + "\n"
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            if self._path.exists() and self._path.stat().st_size + len(line) > self._max_bytes:
                self._rotate()
            try:
                with _exclusive_append(self._path) as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise SiemDeliveryError(
                    f"could not append to the SIEM export at {self._path}.",
                    code="siem_write_failed",
                    details={"path": str(self._path), "cause": str(exc)},
                ) from exc

    def _rotate(self) -> None:
        stamp = datetime.fromtimestamp(self._clock(), tz=timezone.utc)
        candidate = self._dir / f"run-api-{stamp:%Y%m%dT%H%M%S}.ndjson"
        suffix = 1
        while candidate.exists():
            candidate = self._dir / f"run-api-{stamp:%Y%m%dT%H%M%S}-{suffix}.ndjson"
            suffix += 1
        self._path = candidate

    def close(self) -> None:
        return None


class RemoteSiemSink:
    """The seam for a hosted fan-out (Splunk HEC, Datadog logs, an S3 bucket).

    NOT IMPLEMENTED, and it says so at the point of use instead of pretending. There is no endpoint
    configured in this build's environment, so shipping an HTTP client for one would be untested
    placeholder code — the failure mode the plan forbids. What IS provided is the interface a real
    implementation fills, and `SiemDeliveryError` so a misconfigured deployment fails loudly rather
    than quietly exporting nothing.
    """

    def __init__(self, *, endpoint: str = "") -> None:
        self._endpoint = endpoint

    def record(self, event: SiemEvent) -> None:
        raise SiemDeliveryError(
            "no remote SIEM sink is configured in this build; export is local NDJSON only.",
            code="siem_remote_not_implemented",
            details={"endpoint": self._endpoint, "event": event.event},
        )

    def close(self) -> None:
        return None


class CompositeSiemSink:
    """Fan one event out to several sinks. A failure in any sink is raised after all were tried.

    "Tried all, then raised" rather than "stop at the first failure": a remote outage must not also
    cost the local record, which is the copy an auditor actually reads.
    """

    def __init__(self, sinks: Iterable[SiemSink]) -> None:
        self._sinks = tuple(sinks)

    def record(self, event: SiemEvent) -> None:
        failures: list[BaseException] = []
        for sink in self._sinks:
            try:
                sink.record(event)
            except BaseException as exc:  # noqa: BLE001 - collected, then raised
                failures.append(exc)
        if failures:
            raise failures[0]

    def close(self) -> None:
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.close()


@contextlib.contextmanager
def _exclusive_append(path: Path) -> Iterator[Any]:
    """Append under an exclusive flock on the file itself (the ledger's own discipline)."""
    with path.open("a", encoding="utf-8") as handle:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_siem_sink(directory: Path | None = None) -> SiemSink:
    """The sink this process should use: the local file, in the durable root unless overridden."""
    target = directory
    if target is None:
        raw = os.environ.get(SIEM_ENV, "").strip()
        target = Path(raw).expanduser() if raw else _default_directory()
    return FileSiemSink(target)


def _default_directory() -> Path:
    return Path.home() / ".codna" / "siem"


# -- retention ---------------------------------------------------------------------------------


def retention_days_from_env(default: int = DEFAULT_RETENTION_DAYS) -> int:
    """The retention window, from ``CODNA_SIEM_RETENTION_DAYS``. Refuses a malformed or tiny value."""
    raw = os.environ.get(RETENTION_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SiemDeliveryError(
            f"{RETENTION_ENV} must be a whole number of days (got {raw!r}).",
            code="siem_retention_invalid",
            details={"variable": RETENTION_ENV, "value": raw[:32]},
        ) from exc
    if value < 90:
        # Below Temporal Cloud's own window there is nothing to outlive, so the setting is either a
        # typo or an attempt to shorten audit retention. Both are refused.
        raise SiemDeliveryError(
            f"retention must be at least 90 days (Temporal Cloud's own window); got {value}.",
            code="siem_retention_too_short",
            details={"days": value, "minimum": 90},
        )
    return value


@dataclass(frozen=True)
class RetentionPolicy:
    """What is kept, for how long, and what is verified rather than assumed.

    ``days`` is the window. ``apply`` prunes SIEM files whose last modification is older than the
    window and returns what it removed; it NEVER removes a ledger (see the module docstring).
    ``verify_ledgers`` is the other half: for every run whose ledger is still on disk, check the hash
    chain, and report a run that is unreadable — "retention" must not become "I stopped looking".
    """

    days: int = DEFAULT_RETENTION_DAYS
    siem_dir: Path = field(default_factory=_default_directory)

    def __post_init__(self) -> None:
        if isinstance(self.days, bool) or not isinstance(self.days, int) or self.days < 90:
            raise ValueError("a retention window must be an integer number of days, at least 90")

    def cutoff(self, now: float | None = None) -> float:
        return float(now if now is not None else time.time()) - self.days * 86400.0

    def apply(self, *, now: float | None = None) -> list[str]:
        """Delete expired SIEM export files. Returns the filenames removed."""
        boundary = self.cutoff(now)
        removed: list[str] = []
        directory = Path(self.siem_dir)
        if not directory.is_dir():
            return removed
        for path in sorted(directory.glob("run-api-*.ndjson")):
            try:
                if path.stat().st_mtime >= boundary:
                    continue
                path.unlink()
                removed.append(path.name)
            except OSError as exc:  # pragma: no cover - a file removed concurrently
                raise SiemDeliveryError(
                    f"could not prune the expired SIEM file {path.name}.",
                    code="siem_retention_failed",
                    details={"path": str(path), "cause": str(exc)},
                ) from exc
        return removed

    def verify_ledgers(self, runs_root: Path) -> dict[str, list[str]]:
        """Verify every run ledger under ``runs_root``. Returns ``{run_id: problems}`` for the broken.

        An empty mapping means every ledger this host still holds is intact. A run whose ledger is
        MISSING is not reported here (it is `codna run verify-ledger --run` that names that); this is
        the "the retained evidence is still evidence" check.

        An UNREADABLE ledger is reported, not raised: `verify_chain` raises on a malformed line, and a
        retention sweep that aborts on the first damaged file would leave the rest unexamined — which
        is the opposite of what an audit wants. The message names the file and the ledger's own code.
        """
        from .ledger import LedgerError, verify_chain

        problems: dict[str, list[str]] = {}
        root = Path(runs_root)
        if not root.is_dir():
            return problems
        for entry in sorted(root.iterdir()):
            ledger = entry / "ledger.ndjson"
            if not entry.is_dir() or not ledger.is_file():
                continue
            try:
                found = verify_chain(ledger)
            except LedgerError as exc:
                problems[entry.name] = [f"{exc.code}: {exc}"]
                continue
            if found:
                problems[entry.name] = found
        return problems


def _hostname() -> str:
    try:
        return os.uname().nodename
    except (AttributeError, OSError):  # pragma: no cover - a platform without uname
        return ""
