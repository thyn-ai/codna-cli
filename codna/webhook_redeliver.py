"""Missed-delivery sweeper: ask GitHub to redeliver App webhook deliveries that failed.

GitHub does not redeliver failed webhook deliveries on its own. Every single-machine deploy of the
webhook replaces the machine for ~40 s, and each delivery GitHub attempted in that window was
refused at the edge and recorded as failed -- the push that moved a head, the `@codna fix` reply.
The App API exposes exactly what is needed: ``GET /app/hook/deliveries`` (cursor-paged, newest
first) lists them with their status, and ``POST /app/hook/deliveries/{id}/attempts`` asks for a
redelivery (202). A redelivery reuses the original ``X-GitHub-Delivery`` GUID, so the queue's
``delivery_id`` UNIQUE makes asking twice harmless.

Behind ``CODNA_WEBHOOK_REDELIVER=1`` (default off). Runs on the reaper/scaler leader every
``interval_s``; only deliveries newer than ``window_s`` are considered, each id is asked at most
once per process, and the sweep stops at the first page older than the window.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

_API = "https://api.github.com"


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook-redeliver", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


def enabled(environ: Any = None) -> bool:
    e = os.environ if environ is None else environ
    return (e.get("CODNA_WEBHOOK_REDELIVER") or "0").strip().lower() in ("1", "true", "yes", "on")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def failed_deliveries(pages: list[list[dict[str, Any]]], *, since: datetime) -> list[dict[str, Any]]:
    """Pure selection over already-fetched pages: original (not redelivery) deliveries newer than
    ``since`` whose status is not OK. Stops at the first delivery older than ``since``."""
    out: list[dict[str, Any]] = []
    for page in pages:
        for d in page:
            at = _parse_time(d.get("delivered_at"))
            if at is not None and at < since:
                return out
            if d.get("redelivery"):
                continue
            code = d.get("status_code")
            ok = (isinstance(code, int) and 200 <= code < 300) or str(d.get("status") or "").upper() == "OK"
            if not ok:
                out.append(d)
    return out


class Sweeper:
    def __init__(self, *, app_id: str | None, private_key: str | None, github: Any = None,
                 window_s: float = 900.0, interval_s: float = 300.0, max_pages: int = 5,
                 clock: Callable[[], datetime] | None = None) -> None:
        self._app_id = app_id
        self._private_key = private_key
        self._github = github
        self._window = timedelta(seconds=float(window_s))
        self._interval = float(interval_s)
        self._max_pages = int(max_pages)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._asked: dict[str, float] = {}
        self._stop = threading.Event()
        self.redelivered_total = 0
        self.errors_total = 0

    def _jwt(self) -> str | None:
        if not (self._app_id and self._private_key):
            return None
        from .webhook_github import _app_jwt

        return _app_jwt(self._app_id, self._private_key, now=int(time.time()))

    def _fetch_pages(self, jwt: str) -> list[list[dict[str, Any]]]:
        import httpx

        headers = {"Authorization": f"Bearer {jwt}", "Accept": "application/vnd.github+json"}
        url = f"{_API}/app/hook/deliveries?per_page=100"
        pages: list[list[dict[str, Any]]] = []
        for _ in range(self._max_pages):
            resp = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
            if resp.status_code >= 300:
                raise RuntimeError(f"deliveries: {resp.status_code}")
            rows = resp.json()
            if not isinstance(rows, list) or not rows:
                break
            pages.append(rows)
            oldest = _parse_time(rows[-1].get("delivered_at"))
            if oldest is not None and oldest < self._clock() - self._window:
                break
            nxt = [part for part in resp.headers.get("link", "").split(",") if 'rel="next"' in part]
            if not nxt:
                break
            url = nxt[0].split(";")[0].strip().strip("<>")
        return pages

    def _redeliver(self, jwt: str, delivery_id: int) -> bool:
        import httpx

        resp = httpx.post(f"{_API}/app/hook/deliveries/{delivery_id}/attempts",
                          headers={"Authorization": f"Bearer {jwt}", "Accept": "application/vnd.github+json"}, timeout=30.0)
        return resp.status_code in (200, 202)

    def sweep(self) -> int:
        """One pass. Returns how many redeliveries were requested."""
        try:
            jwt = self._jwt()
            if not jwt:
                return 0
            pages = self._github.fetch_pages(jwt) if self._github is not None else self._fetch_pages(jwt)
        except Exception as exc:  # noqa: BLE001
            self.errors_total += 1
            _log("sweep_error", error=type(exc).__name__)
            return 0
        since = self._clock() - self._window
        asked = 0
        now = time.monotonic()
        for d in failed_deliveries(pages, since=since):
            key = str(d.get("id"))
            if key in self._asked:
                continue
            try:
                ok = self._github.redeliver(jwt, int(d["id"])) if self._github is not None else self._redeliver(jwt, int(d["id"]))
            except Exception as exc:  # noqa: BLE001
                self.errors_total += 1
                _log("redeliver_error", delivery=key, error=type(exc).__name__)
                continue
            self._asked[key] = now
            if ok:
                asked += 1
                _log("redelivered", delivery=key, guid=d.get("guid"), github_event=d.get("event"),
                     action=d.get("action"), status_code=d.get("status_code"))
        # forget ids older than a day so the map stays bounded
        self._asked = {k: t for k, t in self._asked.items() if now - t < 86400}
        self.redelivered_total += asked
        return asked

    def start(self) -> None:
        threading.Thread(target=self._loop, name="codna-webhook-redeliver", daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.sweep()
            except Exception as exc:  # noqa: BLE001
                _log("sweep_loop_error", error=type(exc).__name__)

    def stop(self) -> None:
        self._stop.set()
