from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    url: str
    status_code: int | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None


def request_json(url: str, *, timeout_s: float) -> ProbeResult:
    return request_json_with_method("GET", url, timeout_s=timeout_s)


def request_json_with_method(
    method: str,
    url: str,
    *,
    timeout_s: float,
    headers: dict[str, str] | None = None,
) -> ProbeResult:
    try:
        response = httpx.request(method, url, timeout=timeout_s, headers=headers)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(ok=False, url=url, error=f"{type(exc).__name__}: {exc}")
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return ProbeResult(
        ok=200 <= response.status_code < 300,
        url=url,
        status_code=response.status_code,
        payload=payload if isinstance(payload, dict) else None,
        error=None if 200 <= response.status_code < 300 else response.text[:200],
    )


def probe_engine_health(base_url: str, *, timeout_s: float) -> ProbeResult:
    return request_json(f"{base_url.rstrip('/')}/v1/health", timeout_s=timeout_s)


def probe_engine_ready(base_url: str, *, timeout_s: float) -> ProbeResult:
    probe = probe_engine_health(base_url, timeout_s=timeout_s)
    if not probe.ok:
        return probe
    payload = dict(probe.payload or {})
    status = payload.get("status")
    if status not in {"ok", "ready"}:
        return ProbeResult(
            ok=False,
            url=probe.url,
            status_code=probe.status_code,
            payload=payload,
            error="engine liveness status is not ok",
        )
    payload.setdefault("service", "codna-engine")
    payload["readiness"] = "local_offline"
    payload["db"] = "not_required"
    return ProbeResult(
        ok=True,
        url=probe.url,
        status_code=probe.status_code,
        payload=payload,
    )


def probe_engine_version(base_url: str, *, timeout_s: float) -> ProbeResult:
    return request_json(f"{base_url.rstrip('/')}/v1/version", timeout_s=timeout_s)


def probe_sidecar_health(base_url: str, *, timeout_s: float) -> ProbeResult:
    return request_json(f"{base_url.rstrip('/')}/health", timeout_s=timeout_s)


def probe_sidecar_ready(base_url: str, *, timeout_s: float) -> ProbeResult:
    return request_json(f"{base_url.rstrip('/')}/ready", timeout_s=timeout_s)


def request_sidecar_restart(
    base_url: str,
    *,
    runtime_id: str,
    timeout_s: float,
) -> ProbeResult:
    return request_json_with_method(
        "POST",
        f"{base_url.rstrip('/')}/admin/restart",
        timeout_s=timeout_s,
        headers={"x-codna-runtime-id": runtime_id},
    )


def request_sidecar_shutdown(
    base_url: str,
    *,
    runtime_id: str,
    timeout_s: float,
) -> ProbeResult:
    return request_json_with_method(
        "POST",
        f"{base_url.rstrip('/')}/admin/shutdown",
        timeout_s=timeout_s,
        headers={"x-codna-runtime-id": runtime_id},
    )


def wait_until_ready(
    probe,
    base_url: str,
    *,
    timeout_s: float,
    request_timeout_s: float,
) -> ProbeResult:
    deadline = time.monotonic() + timeout_s
    last = ProbeResult(ok=False, url=base_url, error="probe not started")
    while time.monotonic() < deadline:
        last = probe(base_url, timeout_s=request_timeout_s)
        if last.ok:
            return last
        time.sleep(0.5)
    return last
