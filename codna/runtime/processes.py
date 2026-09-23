from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import shutil
import subprocess
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def runtime_lock(lock_path: Path, *, timeout_s: float) -> Iterator[None]:
    ensure_directories(lock_path.parent)
    with lock_path.open("a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for runtime lock {lock_path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: Path, payload: dict) -> None:
    ensure_directories(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    fsync_directory(path.parent)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def rotate_log(path: Path, *, max_bytes: int, backups: int) -> None:
    ensure_directories(path.parent)
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    oldest = path.with_name(f"{path.name}.{backups}")
    oldest.unlink(missing_ok=True)
    for index in range(backups - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if source.exists():
            source.replace(path.with_name(f"{path.name}.{index + 1}"))
    path.replace(path.with_name(f"{path.name}.1"))


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_create_time(pid: int) -> float | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, PermissionError):
        return None
    if completed.returncode != 0:
        return None
    value = " ".join(completed.stdout.strip().split())
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError:
        return None


def process_command(pid: int) -> str | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, PermissionError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _send_signal_with_fallback(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
        return
    except ProcessLookupError:
        raise
    except PermissionError:
        kill_binary = shutil.which("kill") or "/bin/kill"
        completed = subprocess.run(
            [kill_binary, f"-{sig.name.removeprefix('SIG')}", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 or not pid_is_alive(pid):
            return
        shell_binary = shutil.which("zsh") or "/bin/sh"
        completed = subprocess.run(
            [shell_binary, "-lc", f"kill -{sig.name.removeprefix('SIG')} {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 or not pid_is_alive(pid):
            return
        raise


def spawn_detached_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[bytes]:
    ensure_directories(log_path.parent)
    with log_path.open("ab") as log_handle:
        return subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
            close_fds=True,
        )


def terminate_pid(pid: int, *, grace_timeout_s: float, force_timeout_s: float) -> None:
    if not pid_is_alive(pid):
        return
    try:
        _send_signal_with_fallback(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_timeout_s
    while time.monotonic() < deadline:
        if not pid_is_alive(pid):
            return
        time.sleep(0.1)
    try:
        _send_signal_with_fallback(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + force_timeout_s
    while time.monotonic() < deadline:
        if not pid_is_alive(pid):
            return
        time.sleep(0.1)
