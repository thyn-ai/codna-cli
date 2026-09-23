"""Process control for webhook jobs: run the codna CLI under a hard bound, kill its whole process
group on timeout, and tear down the detached runtime each job leaves behind. Split out of
webhook_worker.py (modularity ceiling); the worker re-exports these names."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

# After the timeout the WHOLE process group is killed and the pipe drain is capped at this. Plain
# subprocess.run() kills only the direct child, so a grandchild still holding stdout/stderr (a
# `git clone` remote helper, a test runner) kept communicate() blocked for as long as it lived --
# the thread stayed alive (so /ready stayed green) while processing nothing. Both worker threads
# wedged that way on 2026-09-17 and every review and fix went silent for hours.
_KILL_DRAIN_S = 15


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # start_new_session=True makes pid == pgid
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def _run_job_process(
    argv: list[str], *, env: dict[str, str] | None, cwd: str | None, timeout: float,
    on_process: "Callable[[subprocess.Popen], None] | None" = None,
) -> subprocess.CompletedProcess:
    """subprocess.run(capture_output=True, timeout=...) with a bounded worst case: on timeout the
    process group dies and the drain itself is capped, so a wedged job costs a worker thread at most
    timeout + _KILL_DRAIN_S. Raises TimeoutExpired exactly like subprocess.run does.

    ``on_process`` is handed the live Popen the moment it exists, so a canceller on another thread
    (a superseding head, the pool's deadline watchdog) can kill this process group instead of
    waiting out ``timeout``. It is called before the first read, and a callback that raises must
    never take the job down -- hygiene cannot be allowed to break the thing it is watching.
    """
    bound = timeout
    proc = subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if on_process is not None:
        try:
            on_process(proc)
        except Exception:  # noqa: BLE001 -- registering for cancellation must never fail the job
            pass
    try:
        stdout, stderr = proc.communicate(timeout=bound)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=_KILL_DRAIN_S)
        except subprocess.TimeoutExpired:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            stdout, stderr = "", ""
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        raise subprocess.TimeoutExpired(argv, bound, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


_PID_KEYS = frozenset({"pid", "listener_pid", "supervisor_pid"})


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pids_from_state_files(runtime_root: Path) -> set[int]:
    """Every pid recorded in the runtime's own state files (local-stack.json, local-agent-core.json,
    launcher state): the format is theirs, so walk any JSON under the root for pid-like keys."""
    pids: set[int] = set()
    if not runtime_root.is_dir():
        return pids
    for path in runtime_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".state"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — not JSON, or being written right now
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in _PID_KEYS and isinstance(value, int) and value > 1:
                        pids.add(value)
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
    return pids


def _pids_with_runtime_root(runtime_root: Path) -> set[int]:
    """Linux: every process whose environment carries this job's CODNA_RUNTIME_ROOT -- the
    detached daemons inherit it, so this finds them even if a state file was never written."""
    needle = f"CODNA_RUNTIME_ROOT={runtime_root}".encode()
    pids: set[int] = set()
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return pids
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if needle in environ.split(b"\0"):
            pids.add(int(entry.name))
    return pids


def _teardown_job_runtime(tmp: str) -> int:
    """Stop every detached process a job's codna run left behind.

    codna starts its agent-core sidecar, engine and local stack with start_new_session=True, so
    killing the CLI's process group never reaches them, and nothing else stops them once the CLI
    exits -- each job leaked a Bun sidecar plus engine on the 2 GB machine. Eight reviews in eight
    minutes were enough on 2026-09-17: the ninth failed with "agent-core failed while waiting for
    sidecar readiness" and the machine never recovered until a restart. Returns how many pids
    were signalled. Never raises.
    """
    root = Path(tmp) / ".codna-runtime"
    try:
        pids = _pids_from_state_files(root) | _pids_with_runtime_root(root)
    except Exception:  # noqa: BLE001
        return 0
    pids.discard(os.getpid())
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(_pid_alive(p) for p in pids):
        time.sleep(0.1)
    for pid in pids:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    if pids:
        print(
            json.dumps(
                {
                    "service": "codna-webhook-worker",
                    "event": "runtime_teardown",
                    "pids": sorted(pids),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    return len(pids)


_SCRATCH_PREFIX = "codna-"


def prepare_scratch_root() -> Path | None:
    """Make ``TMPDIR`` this process's default temp dir and clear stale codna scratch from it.

    On the hosted webhook ``TMPDIR`` points at the mounted volume (``/data/tmp`` in
    ``infra/fly/codna-webhook.fly.toml``). The machine's 7.8 GB root filesystem was found 6.5 GB
    full of ephemeral data no ``du`` could see (a restart cleared it; the image itself is small),
    and one review of a large repository needs ~1.1 GB of scratch (its clone, the checkout, the
    agent's worktree) -- reviews died there with ENOSPC (thyn-ai/algenta#1023).
    Everything a job creates lands under this root: the worker's per-job directory is made here
    and the job's own TMPDIR points inside it. Every ``codna-*`` directory still present when the
    process starts belongs to a job that no longer exists (a previous process killed mid-job, or an
    older CLI's leaked review clone), so it is removed; entries with any other name are left
    alone. No-op when ``TMPDIR`` is unset.
    """
    raw = os.environ.get("TMPDIR")
    if not raw:
        return None
    root = Path(raw)
    root.mkdir(parents=True, exist_ok=True)
    import tempfile

    tempfile.tempdir = str(root)  # the documented override for every tempfile call's default dir
    removed = 0
    for entry in root.iterdir():
        if entry.name.startswith(_SCRATCH_PREFIX) and not entry.is_symlink() and entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    if removed:
        print(json.dumps({"event": "scratch_sweep", "root": str(root), "removed": removed}),
              file=sys.stderr)
    return root
