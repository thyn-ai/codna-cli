from __future__ import annotations

from pathlib import Path
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Iterable

LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}
LSOF_FALLBACK_PATHS = (Path("/usr/sbin/lsof"), Path("/usr/bin/lsof"))


def _port_inspection_timeout_s() -> float:
    """Timeout for the `lsof` port probe. The old 2s default was too tight on a busy machine (notably
    macOS, where `lsof` can stall for seconds under load), causing spurious `port_inspection_timeout`
    failures even when the sidecar was up and listening. Default to 10s; allow override for slow hosts."""
    raw = os.environ.get("CODNA_PORT_INSPECTION_TIMEOUT_S")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return 10.0


PORT_INSPECTION_TIMEOUT_S = _port_inspection_timeout_s()
LOOPBACK_CONNECT_TIMEOUT_S = 0.25


class PortInspectionError(RuntimeError):
    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class ListenerInfo:
    pid: int | None
    command: str | None
    host: str
    port: int
    raw_name: str
    parent_pid: int | None = None

    @property
    def loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS


def _parse_host_port(raw_name: str) -> tuple[str, int] | None:
    name = raw_name.strip()
    if name.startswith("[") and "]:" in name:
        host, _, port = name[1:].partition("]:")
        if port.isdigit():
            return host, int(port)
        return None
    if ":" not in name:
        return None
    host, _, port = name.rpartition(":")
    if not port.isdigit():
        return None
    return host, int(port)


def _parse_lsof_fields(lines: Iterable[str]) -> list[ListenerInfo]:
    listeners: list[ListenerInfo] = []
    current: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        key = line[0]
        value = line[1:]
        if key == "p":
            if current:
                parsed = _listener_from_fields(current)
                if parsed:
                    listeners.append(parsed)
            current = {"p": value}
            continue
        current[key] = value
    if current:
        parsed = _listener_from_fields(current)
        if parsed:
            listeners.append(parsed)
    return listeners


def _listener_from_fields(fields: dict[str, str]) -> ListenerInfo | None:
    pid = int(fields["p"]) if fields.get("p", "").isdigit() else None
    parent_pid = int(fields["R"]) if fields.get("R", "").isdigit() else None
    command = fields.get("c")
    name = fields.get("n")
    if not name:
        return None
    host_port = _parse_host_port(name)
    if host_port is None:
        return None
    host, port = host_port
    return ListenerInfo(
        pid=pid,
        command=command,
        host=host,
        port=port,
        raw_name=name,
        parent_pid=parent_pid,
    )


def _loopback_port_accepts_connection(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=LOOPBACK_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


def _lsof_binary() -> str:
    resolved = shutil.which("lsof")
    if resolved:
        return resolved
    for candidate in LSOF_FALLBACK_PATHS:
        if candidate.is_file():
            return str(candidate)
    raise PortInspectionError(
        "Codna cannot inspect local runtime ports because lsof is not installed.",
        code="port_inspection_unavailable",
        details={
            "required_binary": "lsof",
            "fallback_paths": [str(path) for path in LSOF_FALLBACK_PATHS],
        },
    )


def listeners_on_port(port: int) -> list[ListenerInfo]:
    command = [
        _lsof_binary(),
        "-nP",
        "-R",
        f"-iTCP:{port}",
        "-sTCP:LISTEN",
        "-FpcnR",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=PORT_INSPECTION_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        if not _loopback_port_accepts_connection(port):
            return []
        raise PortInspectionError(
            f"Timed out while inspecting live port {port}.",
            code="port_inspection_timeout",
            details={
                "port": port,
                "timeout_seconds": PORT_INSPECTION_TIMEOUT_S,
                "command": command,
            },
        ) from exc
    except OSError as exc:
        raise PortInspectionError(
            f"Could not run lsof while inspecting port {port}.",
            code="port_inspection_failed",
            details={"port": port, "error": str(exc), "command": command},
        ) from exc
    if completed.returncode not in {0, 1}:
        raise PortInspectionError(
            f"lsof failed while inspecting port {port}.",
            code="port_inspection_failed",
            details={
                "port": port,
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
                "stdout": completed.stdout.strip(),
            },
        )
    return _parse_lsof_fields(completed.stdout.splitlines())
