"""Compatibility wrapper for the owned fixed local runtime manager.

The normal user path is `codna fix` / `codna doctor --start-stack`. This module remains as a
dev-facing shell entrypoint so existing wrappers can start, inspect, or stop the same runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from .runtime import LocalRuntimeError, ensure_running, inspect_runtime, stop_runtime
from .runtime.config import DEFAULT_PORT_BASE

STATE_PATH = Path(os.environ.get("CODNA_LOCAL_STACK_STATE") or (Path.home() / ".codna" / "runtime" / "local-stack.json"))
LEGACY_STATE_PATH = Path(os.environ.get("CODNA_LAUNCHER_STATE") or (Path.home() / ".codna" / "launcher.state"))


def resolve_port_base(value: str | None = None) -> int:
    raw = value if value is not None else os.environ.get("CODNA_PORT_BASE")
    if raw in (None, ""):
        return DEFAULT_PORT_BASE
    try:
        port_base = int(raw)
    except ValueError as exc:
        raise ValueError("CODNA_PORT_BASE must be an integer.") from exc
    if port_base < 1024 or port_base > 65534:
        raise ValueError("CODNA_PORT_BASE must be between 1024 and 65534.")
    return port_base


def local_runtime_urls(port_base: int | None = None) -> tuple[str, str]:
    base = resolve_port_base(None if port_base is None else str(port_base))
    return f"http://127.0.0.1:{base}", f"http://127.0.0.1:{base + 1}"


def read_state() -> dict | None:
    status = inspect_runtime()
    return status.get("state") if status.get("local") else None


def start_detached() -> dict:
    endpoint = ensure_running()
    payload = inspect_runtime()
    payload["started"] = {
        "engine_url": endpoint.engine_url,
        "sidecar_url": endpoint.sidecar_url,
        "local": endpoint.local,
        "port_base": endpoint.port_base,
        "runtime_id": endpoint.runtime_id,
    }
    return payload


def stop_running() -> dict:
    return stop_runtime()


def status_payload() -> dict:
    return inspect_runtime()


def serve() -> int:
    start_detached()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m codna.supervisor",
        description="Manage the local Codna engine + sidecar supervisor.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the managed stack in the foreground.")
    subparsers.add_parser("start", help="Start the managed stack in the background.")
    subparsers.add_parser("stop", help="Stop the managed background stack.")
    subparsers.add_parser("status", help="Print the managed stack state.")
    parser.set_defaults(command="serve")
    args = parser.parse_args(argv)

    if args.command == "serve":
        return serve()
    if args.command == "start":
        try:
            print(json.dumps(start_detached(), indent=2))
            return 0
        except LocalRuntimeError as exc:
            print(
                json.dumps(exc.to_dict(), indent=2),
                file=sys.stderr,
            )
            return 1
    if args.command == "stop":
        try:
            payload = stop_running()
            print(json.dumps(payload, indent=2))
            return 0
        except LocalRuntimeError as exc:
            print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
            return 1
    print(json.dumps(status_payload(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
