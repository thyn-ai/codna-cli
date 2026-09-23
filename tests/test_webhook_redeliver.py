"""The missed-delivery sweeper (codna.webhook_redeliver): off by default, selects only failed
original deliveries inside the window, asks GitHub once per delivery, and never raises."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from codna import webhook_redeliver as wr

T0 = datetime(2026, 9, 20, 3, 0, tzinfo=timezone.utc)


def _d(i, *, minutes_ago, status_code=502, redelivery=False):
    return {"id": i, "guid": f"g{i}", "event": "pull_request", "action": "synchronize",
            "status": "OK" if 200 <= status_code < 300 else "Invalid HTTP Response: 502",
            "status_code": status_code, "redelivery": redelivery,
            "delivered_at": (T0 - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")}


def test_enabled_only_when_asked():
    assert wr.enabled({}) is False
    assert wr.enabled({"CODNA_WEBHOOK_REDELIVER": "1"}) is True
    assert wr.enabled({"CODNA_WEBHOOK_REDELIVER": "TRUE"}) is True
    assert wr.enabled({"CODNA_WEBHOOK_REDELIVER": "no"}) is False


def test_failed_deliveries_selects_failed_originals_inside_the_window_and_stops_at_the_edge():
    pages = [[_d(1, minutes_ago=1, status_code=202), _d(2, minutes_ago=2), _d(3, minutes_ago=3, redelivery=True),
              _d(4, minutes_ago=4, status_code=503)],
             [_d(5, minutes_ago=20), _d(6, minutes_ago=30)]]                      # older than the window: never reached
    picked = wr.failed_deliveries(pages, since=T0 - timedelta(minutes=15))
    assert [d["id"] for d in picked] == [2, 4]                                   # OK skipped, redelivery skipped, old cut off


class _Api:
    def __init__(self, pages, *, fail=False):
        self.pages = pages
        self.fail = fail
        self.redelivered = []

    def fetch_pages(self, jwt):
        if self.fail:
            raise RuntimeError("deliveries: 503")
        return self.pages

    def redeliver(self, jwt, delivery_id):
        self.redelivered.append(delivery_id)
        return True


def _sweeper(api, **kw):
    s = wr.Sweeper(app_id="1", private_key="pem", github=api, window_s=900, clock=lambda: T0, **kw)
    s._jwt = lambda: "jwt"  # the App JWT needs a real key; the fake API never reads it
    return s


def test_sweeper_asks_once_per_failed_delivery_and_counts():
    api = _Api([[_d(1, minutes_ago=1), _d(2, minutes_ago=2, status_code=200), _d(3, minutes_ago=3)]])
    s = _sweeper(api)
    assert s.sweep() == 2 and api.redelivered == [1, 3]
    assert s.sweep() == 0 and api.redelivered == [1, 3]                          # remembered: not asked twice
    assert s.redelivered_total == 2 and s.errors_total == 0


def test_sweeper_swallows_api_failures():
    s = _sweeper(_Api([], fail=True))
    assert s.sweep() == 0 and s.errors_total == 1


def test_sweeper_without_app_credentials_does_nothing():
    s = wr.Sweeper(app_id=None, private_key=None, github=_Api([[_d(1, minutes_ago=1)]]))
    assert s.sweep() == 0
