"""The local Temporal dev server: find it, start it, probe it, stop it — never guess.

``codna durable`` needs a Temporal server. Locally that is ``temporal server start-dev``: a single
binary with an embedded SQLite database, which is why the durable extra does not ship a server of
its own. This module owns the whole lifecycle of that process and nothing else — the workflows, the
activities and the worker are deliberately absent, so a failure to start the server can never be
confused with a failure to run a workflow.

Four rules, all of them the consequence of a real failure mode:

* **Absence fails loudly, with an install hint.** A missing binary raises
  ``dev_server_binary_missing`` naming the two places we looked; it never falls back to "the worker
  will probably find a server somewhere".
* **A managed process is identifiable.** Starting writes ``dev-server.json`` (pid, address,
  database) beside the database. A port that is LISTENING without that file belongs to someone else,
  and is reported as ``dev_server_foreign`` rather than adopted — killing a stranger's server (or
  reusing it with the wrong namespace) is how a run disappears into the wrong cluster.
* **Readiness is a TCP connect, not a sleep.** ``start-dev`` prints its banner before it accepts
  gRPC, so waiting on the log is unreliable; the poll is the same probe the CLI uses before it
  reports a run started.
* **A crash is reported with its own output.** A spawned server that exits during startup raises
  ``dev_server_exited`` carrying the exit code and the tail of the log, because "timed out waiting
  for the dev server" with no output is a support ticket instead of a diagnosis.

* **Two starts cannot race into two servers.** ``ensure_started`` runs its check-then-spawn inside a
  file lock beside the pid file. Two ``codna durable run`` invocations at the same instant used to be
  able to both probe "down" and both spawn ``start-dev`` on one SQLite database; the second either
  crashes the first or leaves an orphan holding the port. The lock makes "is it up? then start it"
  one atomic step per machine, and the second caller adopts the server the first one started.

No shell is used anywhere: the argv is built explicitly, so a path with a space (or a
``CODNA_TEMPORAL_ADDRESS`` from an env file) cannot become a second command.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, Mapping, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module importable in the base CLI
    from .config import DurableConfig

__all__ = [
    "DEFAULT_PROBE_TIMEOUT_S",
    "DEFAULT_START_TIMEOUT_S",
    "DEFAULT_STOP_TIMEOUT_S",
    "DevServer",
    "DevServerError",
    "DevServerSettings",
    "DevServerStatus",
    "SpawnedProcess",
]

# A TCP connect to loopback is either immediate or refused; 250 ms is generous for "immediate".
DEFAULT_PROBE_TIMEOUT_S: Final = 0.25
# `start-dev` creates/migrates its SQLite database on boot, which is slow the first time.
DEFAULT_START_TIMEOUT_S: Final = 60.0
# SIGTERM first, and the same window for SIGKILL to take effect.
DEFAULT_STOP_TIMEOUT_S: Final = 15.0
# How often the readiness loop probes. Short enough that a normal start feels immediate.
POLL_INTERVAL_S: Final = 0.1
# Lines of the dev-server log quoted in an error. Enough to name the cause, bounded for JSON output.
LOG_TAIL_LINES: Final = 20

LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})

INSTALL_HINT: Final = (
    "install the Temporal CLI (https://docs.temporal.io/cli) or point CODNA_TEMPORAL_ADDRESS at a "
    "running server"
)

PID_FILE_NAME: Final = "dev-server.json"
LOG_FILE_NAME: Final = "temporal-dev.log"
# Serialises "check whether it is up, then start it" across processes. A separate file from the pid
# file so holding the lock never implies the server is running.
LOCK_FILE_NAME: Final = "dev-server.lock"
# How long a second caller waits for the first to finish starting. Above the start timeout, because
# the first caller's whole start (download + SQLite migration) happens while this lock is held.
LOCK_TIMEOUT_S: Final = 120.0
# Deployment versioning is what makes "a new build never steals an in-flight run" enforceable, and a
# dev server does not enable it by default. Passed on every start so the worker's PINNED registration
# cannot fail against a server started by an older command.
DEPLOYMENT_VERSIONS_CONFIG: Final = "matching.enableDeploymentVersions=true"

# The Temporal CLI this build runs, pinned by version AND sha256 in `temporal_cli.lock.json` beside
# this module. A download is verified against the recorded digest before it is trusted, because the
# dev server is a binary we execute with the user's privileges: "download the latest CLI" is a supply
# chain we would be trusting every launch. An unpinned/unknown digest is a REFUSAL, never a warning.
LOCK_FILE_NAME: Final = "temporal_cli.lock.json"
# A download is a whole CLI archive; the timeout bounds a hung connection, not a slow link.
DEFAULT_DOWNLOAD_TIMEOUT_S: Final = 300.0


def default_lock_path() -> Path:
    """The shipped pin, beside this module (so it travels with the installed package)."""
    return Path(__file__).resolve().parent / LOCK_FILE_NAME


def _cli_version() -> str:
    """This CLI's version, for the download's User-Agent. Imported lazily: it is a leaf module."""
    from .. import __version__

    return __version__


class DevServerError(RuntimeError):
    """The dev server could not be found, started, probed or stopped.

    Always carries a stable ``code`` and a machine-readable ``details`` map, plus the same
    ``hint``/``as_dict`` shape the rest of the durable surface uses, so the CLI can print one
    consistent error body.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, object] | None = None,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})
        self.hint = hint

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }
        if self.hint:
            body["hint"] = self.hint
        return {"error": body}


class SpawnedProcess(Protocol):
    """The part of ``subprocess.Popen`` this module needs, so a test can supply a fake."""

    pid: int

    def poll(self) -> int | None: ...


@dataclass(frozen=True)
class DevServerSettings:
    """Everything that locates the dev server: where it listens, and the files it owns.

    Derived from `DurableConfig` for a real command, and constructed directly by tests — which is
    why it exists instead of taking the whole (large, immutable) config as an argument.
    """

    host: str
    port: int
    namespace: str
    db_path: Path
    bin_dir: Path
    log_path: Path
    pid_path: Path
    # The pin: version + sha256 per platform. A field (not a module constant) so a test can point at
    # its own lock file and exercise every refusal without touching the shipped one.
    lock_path: Path | None = None

    @classmethod
    def from_config(cls, config: "DurableConfig") -> "DevServerSettings":
        return cls(
            host=config.temporal_host,
            port=config.temporal_port,
            namespace=config.namespace,
            db_path=config.paths.dev_db_path,
            bin_dir=config.paths.dev_server_bin_dir,
            log_path=config.paths.logs_dir / LOG_FILE_NAME,
            pid_path=config.paths.temporal_dir / PID_FILE_NAME,
            lock_path=default_lock_path(),
        )

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS


@dataclass(frozen=True)
class DevServerStatus:
    """What is actually true about the dev server right now — observed, never assumed."""

    running: bool
    address: str
    pid: int | None
    managed: bool
    stale_pid_file: bool
    db_path: str
    log_path: str
    # True when this call found an already-running server instead of starting one. Reported so a
    # caller can tell "I started it" from "someone else had it up" without guessing from timing.
    adopted: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "running": self.running,
            "address": self.address,
            "pid": self.pid,
            "managed": self.managed,
            "stale_pid_file": self.stale_pid_file,
            "db_path": self.db_path,
            "log_path": self.log_path,
            "adopted": self.adopted,
        }


def _default_spawn(argv: Sequence[str], log_path: Path) -> SpawnedProcess:
    """Start the server detached, with its output in ``log_path``.

    A new session on POSIX so the dev server outlives the CLI invocation that started it (the
    worker connects to it afterwards), and ``stdin=DEVNULL`` so it can never block on a terminal
    read. The log handle is closed here; the child keeps its own descriptor.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    try:
        process = subprocess.Popen(  # noqa: S603 - argv is built here, never a shell string
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        handle.close()
    return process


class DevServer:
    """Lifecycle of one dev server, identified by `DevServerSettings`.

    Every external effect is injectable (``which``, ``spawn``, ``probe``, ``clock``, ``sleep``) so
    the start/stop state machine — which is where the bugs are — is testable without a binary, a
    port or a fixture that waits.
    """

    def __init__(
        self,
        settings: DevServerSettings,
        *,
        binary: str | Path | None = None,
        which: Callable[[str], str | None] = shutil.which,
        spawn: Callable[[Sequence[str], Path], SpawnedProcess] | None = None,
        probe: Callable[[str, int], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        trust_binary: bool = False,
        allow_download: bool = True,
    ) -> None:
        if not 1 <= settings.port <= 65535:
            raise DevServerError(
                f"dev server port must be between 1 and 65535 (got {settings.port}).",
                code="dev_server_port_invalid",
                details={"port": settings.port},
            )
        self.settings = settings
        self._explicit_binary = Path(binary) if binary is not None else None
        self._which = which
        # An explicitly supplied binary is verified against the pin by default. `trust_binary` exists
        # for a test that supplies a stand-in script (and says so), never for production: an unverified
        # binary we then execute is the hole the pin closes.
        self._trust_binary = trust_binary
        self._allow_download = allow_download
        self._spawn = spawn or _default_spawn
        self._probe_fn = probe or self._tcp_probe
        self._clock = clock
        self._sleep = sleep
        self._probe_timeout_s = probe_timeout_s
        self._last_process: SpawnedProcess | None = None

    # -- discovery ------------------------------------------------------------------

    def binary(self) -> str:
        """The pinned Temporal CLI, verified (and downloaded once if absent).

        No ``PATH`` fallback. A developer's own ``temporal`` is a different build with different flags
        and different versioning defaults, so "it worked on my machine" would be the norm — which is
        the exact class of bug the pin exists to prevent. A caller that really wants a different
        binary passes it explicitly (and it is still checked against the pin), or points
        ``CODNA_TEMPORAL_ADDRESS`` at a server they manage.
        """
        return str(self.ensure_binary())

    @staticmethod
    def _executable(path: Path) -> bool:
        return path.is_file() and os.access(path, os.X_OK)

    # -- the pinned CLI ---------------------------------------------------------------

    def lock(self) -> dict[str, Any]:
        """The pin: version plus the sha256 of this platform's binary, or a structured refusal.

        Read rather than embedded so a version bump is one file and one reviewable diff. A malformed
        or partial lock is an error, not "no pin": silently treating an unreadable pin as "install
        whatever" is exactly the supply-chain hole the pin exists to close.
        """
        path = self.settings.lock_path or default_lock_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DevServerError(
                f"the Temporal CLI pin is missing ({path}).",
                code="dev_server_pin_missing",
                details={"path": str(path)},
                hint="reinstall codna, or restore cli/codna/durable/temporal_cli.lock.json",
            ) from exc
        except json.JSONDecodeError as exc:
            raise DevServerError(
                f"the Temporal CLI pin is not valid JSON ({path}).",
                code="dev_server_pin_corrupt",
                details={"path": str(path)},
            ) from exc
        if not isinstance(payload, dict):
            raise DevServerError(
                f"the Temporal CLI pin must be an object ({path}).",
                code="dev_server_pin_corrupt",
                details={"path": str(path), "type": type(payload).__name__},
            )
        entry = self.platform_entry(payload)
        return {
            "cli_version": str(payload.get("cli_version") or ""),
            "archive_url": str(payload.get("archive_url") or ""),
            **entry,
            "path": str(path),
        }

    @staticmethod
    def platform_key() -> str:
        """``<platform>-<arch>`` for the running interpreter, in the lock file's vocabulary."""
        import platform as _platform

        system = sys.platform
        if system.startswith("linux"):
            plat = "linux"
        elif system == "darwin":
            plat = "darwin"
        else:
            plat = system
        machine = _platform.machine().lower()
        arch = {
            "aarch64": "arm64",
            "arm64": "arm64",
            "x86_64": "amd64",
            "amd64": "amd64",
        }.get(machine, machine)
        return f"{plat}-{arch}"

    def platform_entry(self, payload: Mapping[str, Any]) -> dict[str, str]:
        """This platform's pinned digests, refusing a platform the lock does not cover.

        A refusal rather than a fallback digest: a platform we have not verified is a binary we
        cannot vouch for, and executing it anyway would make the pin decorative.
        """
        platforms = payload.get("platforms")
        key = self.platform_key()
        entry = platforms.get(key) if isinstance(platforms, Mapping) else None
        if not isinstance(entry, Mapping):
            raise DevServerError(
                f"the Temporal CLI pin does not cover this platform ({key}).",
                code="dev_server_platform_unsupported",
                details={
                    "platform": key,
                    "supported": sorted(platforms) if isinstance(platforms, Mapping) else [],
                },
                hint="run against a real Temporal server with CODNA_TEMPORAL_ADDRESS",
            )
        digest = entry.get("binary_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DevServerError(
                f"the Temporal CLI pin has no usable sha256 for {key}.",
                code="dev_server_pin_corrupt",
                details={"platform": key, "binary_sha256": repr(digest)},
            )
        return {
            "platform": str(entry.get("platform") or ""),
            "arch": str(entry.get("arch") or ""),
            "binary_sha256": digest.lower(),
            "archive_sha256": str(entry.get("archive_sha256") or "").lower(),
        }

    def managed_binary(self) -> Path:
        """Where the pinned binary lives. ``temporal`` so `binary()` and a manual run agree."""
        return self.settings.bin_dir / "temporal"

    def verify(self, path: Path, *, expected_sha256: str) -> str:
        """The file's sha256, or a refusal naming both digests. Never returns a mismatched value."""
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise DevServerError(
                f"could not read the Temporal CLI at {path}: {exc}",
                code="dev_server_binary_unreadable",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise DevServerError(
                f"the Temporal CLI at {path} does not match the pinned digest.",
                code="dev_server_binary_digest_mismatch",
                details={"path": str(path), "expected_sha256": expected_sha256, "actual_sha256": actual},
                hint="delete it and re-run so it is downloaded again, or reinstall codna",
            )
        return actual

    def ensure_binary(self, *, timeout_s: float = DEFAULT_DOWNLOAD_TIMEOUT_S) -> Path:
        """The verified pinned binary, downloading it once if it is not there yet.

        Order: an explicitly supplied binary (verified against the pin unless ``trust_binary``), the
        managed copy (ALWAYS re-verified — a file on disk is not evidence it is the file we pinned),
        then a download into the managed location. ``PATH`` is deliberately not consulted: a
        developer's own ``temporal`` is a different build, and "works on my machine" is exactly the
        class of bug a pin prevents.
        """
        pin = self.lock()
        expected = pin["binary_sha256"]
        if self._explicit_binary is not None:
            if not self._executable(self._explicit_binary):
                raise DevServerError(
                    f"the Temporal CLI path is not an executable file: {self._explicit_binary}",
                    code="dev_server_binary_missing",
                    details={"path": str(self._explicit_binary)},
                )
            if self._trust_binary:
                return self._explicit_binary
            self.verify(self._explicit_binary, expected_sha256=expected)
            return self._explicit_binary

        managed = self.managed_binary()
        if self._executable(managed):
            self.verify(managed, expected_sha256=expected)
            return managed
        if not self._allow_download:
            raise DevServerError(
                "the pinned Temporal CLI is not installed, and downloading is disabled.",
                code="dev_server_binary_missing",
                details={
                    "path": str(managed),
                    "cli_version": pin["cli_version"],
                    "binary_sha256": expected,
                },
                hint="run `codna durable dev-server install`, or set CODNA_TEMPORAL_ADDRESS "
                "to a server you manage",
            )
        return self._download(pin, timeout_s=timeout_s)

    def _download(self, pin: Mapping[str, str], *, timeout_s: float) -> Path:
        """Download the pinned archive, verify it, and install the binary atomically.

        Verified BEFORE it is made executable or moved into place: a partially-written or tampered
        download must never exist at a path `binary()` would accept, not even briefly.
        """
        url = pin["archive_url"]
        if not url:
            raise DevServerError(
                "the Temporal CLI pin records no download URL.",
                code="dev_server_pin_corrupt",
                details={"path": pin["path"]},
            )
        query = f"platform={pin['platform']}&arch={pin['arch']}"
        separator = "&" if "?" in url else "?"
        full_url = f"{url}{separator}{query}"
        staging = self.settings.bin_dir / ".staging"
        staging.mkdir(parents=True, exist_ok=True)
        archive = staging / f"temporal-{pin['cli_version']}-{self.platform_key()}.tar.gz"
        try:
            self._fetch(full_url, archive, timeout_s=timeout_s)
            if pin["archive_sha256"]:
                self.verify(archive, expected_sha256=pin["archive_sha256"])
            extracted = self._extract(archive, staging)
            self.verify(extracted, expected_sha256=pin["binary_sha256"])
            extracted.chmod(0o755)
            target = self.managed_binary()
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(extracted, target)
        except (OSError, subprocess.SubprocessError) as exc:
            raise DevServerError(
                f"could not install the Temporal CLI from {full_url}: {exc}",
                code="dev_server_download_failed",
                details={"url": full_url, "archive": str(archive), "error": str(exc)},
                hint="check network access, or set CODNA_TEMPORAL_ADDRESS to a server you manage",
            ) from exc
        return target

    def _fetch(self, url: str, destination: Path, *, timeout_s: float) -> None:
        """Fetch ``url`` into ``destination`` with an explicit timeout.

        ``urllib`` rather than a shelled curl: no shell, no argument quoting, and the failure is an
        exception with a reason instead of a non-zero exit whose stderr nobody captured.
        """
        import urllib.error
        import urllib.request

        # An explicit User-Agent: the archive host answers 403 to urllib's default (`Python-urllib/x`),
        # measured. Naming the client also makes our traffic identifiable in the host's logs.
        request = urllib.request.Request(  # noqa: S310 - the URL comes from the pinned lock file
            url,
            headers={"User-Agent": f"codna-durable/{_cli_version()}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                with destination.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DevServerError(
                f"downloading the Temporal CLI failed: {exc}",
                code="dev_server_download_failed",
                details={"url": url, "error": str(exc)},
                hint="check network access, or set CODNA_TEMPORAL_ADDRESS to a server you manage",
            ) from exc

    def _extract(self, archive: Path, staging: Path) -> Path:
        """Extract the ``temporal`` member of the pinned archive, refusing an unexpected layout.

        Written explicitly rather than with ``extractall``: a tarball is untrusted input, and
        ``extractall`` honours member paths (an absolute or ``..`` name would escape the directory).
        Only the one member we want is read, and only into ``staging``.
        """
        import tarfile

        try:
            with tarfile.open(archive, "r:gz") as tar:
                members = [item for item in tar.getmembers() if item.name in ("temporal", "./temporal")]
                if len(members) != 1 or not members[0].isfile():
                    raise DevServerError(
                        "the Temporal CLI archive does not contain exactly one `temporal` binary.",
                        code="dev_server_archive_unexpected",
                        details={
                            "archive": str(archive),
                            "members": [item.name for item in tar.getmembers()][:20],
                        },
                    )
                source = tar.extractfile(members[0])
                if source is None:  # pragma: no cover - isfile() above makes this unreachable
                    raise DevServerError(
                        "the Temporal CLI archive member could not be read.",
                        code="dev_server_archive_unexpected",
                        details={"archive": str(archive)},
                    )
                target = staging / "temporal"
                with source, target.open("wb") as handle:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        handle.write(chunk)
                return target
        except tarfile.TarError as exc:
            raise DevServerError(
                f"the Temporal CLI archive is not readable: {exc}",
                code="dev_server_archive_unreadable",
                details={"archive": str(archive), "error": str(exc)},
            ) from exc

    # -- probing --------------------------------------------------------------------

    def probe(self) -> bool:
        """Whether something is accepting TCP connections at the configured address."""
        return bool(self._probe_fn(self.settings.host, self.settings.port))

    def _tcp_probe(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=self._probe_timeout_s):
                return True
        except OSError:
            return False

    # -- pid file -------------------------------------------------------------------

    def read_pid(self) -> int | None:
        """The pid of the server WE started, or ``None`` when there is no pid file.

        A pid file that exists but cannot be parsed is an error, not ``None``: it means a previous
        version wrote something we do not understand, and silently treating the server as unmanaged
        would leave an orphan process holding the port.
        """
        path = self.settings.pid_path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DevServerError(
                f"the dev-server pid file {path} is not valid JSON.",
                code="dev_server_pid_corrupt",
                details={"path": str(path)},
            ) from exc
        pid = payload.get("pid") if isinstance(payload, dict) else None
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise DevServerError(
                f"the dev-server pid file {path} has no usable pid.",
                code="dev_server_pid_corrupt",
                details={"path": str(path), "pid": repr(pid)},
            )
        return pid

    def _write_pid(self, pid: int) -> None:
        path = self.settings.pid_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": pid,
            "address": self.settings.address,
            "namespace": self.settings.namespace,
            "db_path": str(self.settings.db_path),
        }
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _clear_pid(self) -> None:
        self.settings.pid_path.unlink(missing_ok=True)

    @staticmethod
    def _alive(pid: int) -> bool:
        """Whether a pid is a live process — with a zombie treated as dead.

        ``kill(0)`` alone is not enough: a child that has exited but not been reaped still answers
        it, so ``stop`` would wait out its full timeout (and then report a failure) for a process
        that is already gone. ``waitpid(WNOHANG)`` reaps a pid THIS process owns; for any other pid
        it raises ``ChildProcessError`` and the portable check decides.
        """
        try:
            reaped, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass  # not our child (the normal case for a second `codna durable dev-server stop`)
        except OSError:  # pragma: no cover - platform without waitpid semantics
            pass
        else:
            if reaped == pid:
                return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    # -- status ---------------------------------------------------------------------

    def status(self) -> DevServerStatus:
        return self._status(adopted=False)

    def _status(self, *, adopted: bool) -> DevServerStatus:
        """Observe the server once. ``adopted`` records whether THIS call started it."""
        running = self.probe()
        pid = self.read_pid()
        managed = bool(running and pid is not None and self._alive(pid))
        return DevServerStatus(
            running=running,
            address=self.settings.address,
            pid=pid if managed else None,
            managed=managed,
            stale_pid_file=pid is not None and not managed,
            db_path=str(self.settings.db_path),
            log_path=str(self.settings.log_path),
            adopted=adopted,
        )

    # -- start ----------------------------------------------------------------------

    def argv(self, *, headless: bool = False) -> list[str]:
        """The exact argv for ``temporal server start-dev``.

        ``--ip`` is deliberately NOT passed: the flags a dev server accepts for its bind address
        differ between CLI versions, and a wrong flag is a startup failure with a misleading error.
        A non-loopback address is refused instead, with the instruction to run a real server.
        """
        if not self.settings.is_loopback:
            raise DevServerError(
                "the local dev server only binds loopback, but the configured Temporal address is "
                f"{self.settings.address}.",
                code="dev_server_address_unsupported",
                details={
                    "address": self.settings.address,
                    "hint": "run a real Temporal server and set CODNA_TEMPORAL_ADDRESS to it",
                },
            )
        argv = [
            self.binary(),
            "server",
            "start-dev",
            "--db-filename",
            str(self.settings.db_path),
            "--port",
            str(self.settings.port),
            "--namespace",
            self.settings.namespace,
            # Deployment versioning ON: without it a `WorkerDeploymentConfig` registration leaves runs
            # stuck (measured: a default server routes new runs to `__unversioned__`, and a PINNED
            # worker's runs never reach it). With it, a build bump registers a NEW version and leaves
            # runs already in flight pinned to the build they started on — the plan's risk mitigation.
            "--dynamic-config-value",
            DEPLOYMENT_VERSIONS_CONFIG,
            "--log-level",
            "warn",
        ]
        if headless:
            argv.append("--headless")
        return argv

    def ensure_started(
        self,
        *,
        timeout_s: float = DEFAULT_START_TIMEOUT_S,
        headless: bool = False,
    ) -> DevServerStatus:
        """Start the dev server unless one is already answering; idempotent by design.

        Idempotence matters because every durable command calls this: the worker, ``run`` and
        ``status`` must all be able to say "make sure a server exists" without racing each other
        into two ``start-dev`` processes on one SQLite file.

        **The check and the spawn happen under one lock.** Without it, two commands starting in the
        same instant both probe "down", both spawn, and the loser either crashes against the winner's
        SQLite lock or survives as an orphan holding a port no pid file names. The second caller
        blocks here, then sees the first caller's server and adopts it (``adopted=True`` in the
        payload), which is the outcome "make sure a server exists" promises.
        """
        if timeout_s <= 0:
            raise DevServerError(
                "the dev-server start timeout must be positive.",
                code="dev_server_timeout_invalid",
                details={"timeout_s": timeout_s},
            )
        if self.probe():
            return self._status(adopted=True)
        with self._start_lock(timeout_s=timeout_s):
            # Re-probed INSIDE the lock: the winner may have finished while this caller waited.
            if self.probe():
                return self._status(adopted=True)
            return self._start_locked(timeout_s=timeout_s, headless=headless)

    @contextlib.contextmanager
    def _start_lock(self, *, timeout_s: float) -> "Any":
        """An exclusive, non-blocking-then-waiting file lock around the start sequence.

        A separate lock file from the pid file: the pid file records WHO owns the server, the lock
        only records that a start is in progress. Blocking with a bounded wait (rather than failing)
        is the right behaviour for the second ``run``: it should wait for the server, not error out
        because another command happened to be starting one.
        """
        path = self.settings.pid_path.with_name(LOCK_FILE_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        wait_budget = max(LOCK_TIMEOUT_S, timeout_s + 5.0)
        handle = path.open("a+", encoding="utf-8")
        try:
            deadline = self._clock() + wait_budget
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self._clock() >= deadline:
                        raise DevServerError(
                            "timed out waiting for another codna command to finish starting the "
                            "dev server.",
                            code="dev_server_start_locked",
                            details={
                                "lock_path": str(path),
                                "timeout_s": wait_budget,
                                "pid_path": str(self.settings.pid_path),
                            },
                            hint=f"inspect {self.settings.log_path}, or remove {path} if no start is running",
                        ) from None
                    self._sleep(POLL_INTERVAL_S)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _start_locked(self, *, timeout_s: float, headless: bool) -> DevServerStatus:
        """The start sequence, with the lock held. Assumes the server is not answering."""
        stale = self.read_pid()
        if stale is not None:
            if self._alive(stale):
                # Process exists but is not accepting connections: starting a second one would
                # fight over the same database file, and killing it might be the user's server.
                raise DevServerError(
                    "a dev-server process is running but not accepting connections.",
                    code="dev_server_unresponsive",
                    details={"pid": stale, "address": self.settings.address},
                    hint=f"inspect {self.settings.log_path}, or stop it with `codna durable dev-server stop`",
                )
            self._clear_pid()

        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = self.argv(headless=headless)
        process = self._spawn(argv, self.settings.log_path)
        self._last_process = process
        self._write_pid(process.pid)
        try:
            self._wait_ready(process, timeout_s=timeout_s)
        except DevServerError:
            self._terminate(process.pid, timeout_s=DEFAULT_STOP_TIMEOUT_S)
            self._clear_pid()
            raise
        return self._status(adopted=False)

    def _wait_ready(self, process: SpawnedProcess, *, timeout_s: float) -> None:
        deadline = self._clock() + timeout_s
        while True:
            if self.probe():
                return
            code = process.poll()
            if code is not None:
                raise DevServerError(
                    f"the Temporal dev server exited with code {code} during startup.",
                    code="dev_server_exited",
                    details={
                        "exit_code": code,
                        "log_path": str(self.settings.log_path),
                        "log_tail": self._tail_log(),
                    },
                    hint=f"inspect {self.settings.log_path}",
                )
            if self._clock() >= deadline:
                raise DevServerError(
                    f"the Temporal dev server did not accept connections on "
                    f"{self.settings.address} within {timeout_s:g}s.",
                    code="dev_server_start_timeout",
                    details={
                        "address": self.settings.address,
                        "timeout_s": timeout_s,
                        "log_path": str(self.settings.log_path),
                        "log_tail": self._tail_log(),
                    },
                    hint=f"inspect {self.settings.log_path}",
                )
            self._sleep(POLL_INTERVAL_S)

    # -- stop -----------------------------------------------------------------------

    def stop(self, *, timeout_s: float = DEFAULT_STOP_TIMEOUT_S) -> DevServerStatus:
        """Stop the server WE started, refusing to touch a server we did not start.

        A port that is answering with no pid file is a foreign server: reporting it as stopped
        would be a lie, and killing it could take down a colleague's cluster, so it is an error
        the caller has to resolve.
        """
        if timeout_s <= 0:
            raise DevServerError(
                "the dev-server stop timeout must be positive.",
                code="dev_server_timeout_invalid",
                details={"timeout_s": timeout_s},
            )
        # The same lock as `ensure_started`: stopping while another command is mid-start would either
        # kill the server it just spawned or leave the pid file describing a process we are deleting.
        with self._start_lock(timeout_s=timeout_s):
            return self._stop_locked(timeout_s=timeout_s)

    def _stop_locked(self, *, timeout_s: float) -> DevServerStatus:
        """The stop sequence, with the start lock held."""
        pid = self.read_pid()
        if pid is None:
            if self.probe():
                raise DevServerError(
                    f"something is serving {self.settings.address}, but it was not started by "
                    "codna (no pid file), so it will not be stopped.",
                    code="dev_server_foreign",
                    details={"address": self.settings.address, "pid_path": str(self.settings.pid_path)},
                    hint="stop it yourself, or set CODNA_TEMPORAL_ADDRESS to a server you manage",
                )
            return self.status()

        was_alive = self._alive(pid)
        if was_alive:
            self._terminate(pid, timeout_s=timeout_s)
        self._clear_pid()
        if self.probe():
            raise DevServerError(
                f"the dev server on {self.settings.address} is still accepting connections after "
                f"pid {pid} was terminated.",
                code="dev_server_stop_timeout",
                details={"address": self.settings.address, "pid": pid, "timeout_s": timeout_s},
            )
        return self._status(adopted=False)

    def _terminate(self, pid: int, *, timeout_s: float) -> None:
        """SIGTERM, then SIGKILL, waiting on the PROCESS — not on the port.

        The port is closed during shutdown while the process may still be flushing its SQLite
        database; returning at that moment would let the next start race a half-written file.
        """
        self._signal(pid, signal.SIGTERM)
        if self._wait_exit(pid, timeout_s=timeout_s):
            return
        self._signal(pid, signal.SIGKILL)
        if self._wait_exit(pid, timeout_s=timeout_s):
            return
        raise DevServerError(
            f"the dev-server process {pid} did not exit within {timeout_s:g}s of SIGKILL.",
            code="dev_server_stop_timeout",
            details={"pid": pid, "timeout_s": timeout_s},
        )

    def _signal(self, pid: int, number: int) -> None:
        try:
            os.kill(pid, number)
        except ProcessLookupError:
            return  # already gone: the goal state, not a failure

    def _wait_exit(self, pid: int, *, timeout_s: float) -> bool:
        deadline = self._clock() + timeout_s
        while self._alive(pid):
            if self._clock() >= deadline:
                return False
            self._sleep(POLL_INTERVAL_S)
        return True

    # -- diagnostics -----------------------------------------------------------------

    def _tail_log(self, lines: int = LOG_TAIL_LINES) -> list[str]:
        """The last ``lines`` of the dev-server log, or an empty list if it is not there yet."""
        try:
            text = self.settings.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]


def _as_mapping(value: Any) -> Mapping[str, Any]:  # pragma: no cover - small typing helper
    return value if isinstance(value, Mapping) else {}
