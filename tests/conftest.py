"""Make the `codna` package importable when running pytest from anywhere.

`codna.sarif` / `codna.findings` are pure-stdlib, so importing the package does not pull
in the engine client — the offline SARIF suite runs without installing the full CLI.
"""
import pathlib
import sys

import pytest

CLI_DIR = pathlib.Path(__file__).resolve().parents[1]  # .../cli
FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001 - pytest hook signature
    """Do not collect fixture repositories as pytest modules.

    Some fixture repos intentionally contain broken source files so Codna can test
    syntax-error handling. They are test data, not test modules.
    """
    path = pathlib.Path(collection_path)
    try:
        path.relative_to(FIXTURES_DIR)
    except ValueError:
        return False
    return True


@pytest.fixture(autouse=True)
def _isolated_review_history(monkeypatch, tmp_path):
    """Every review turn records its observed duration and every review budget reads the window
    (codna.review_budget, ``CODNA_REVIEW_HISTORY_DIR`` else ``~/.codna/review-budget``). Tests must
    neither learn from nor teach the developer's real window, so each test gets an empty one. The
    in-process decision cache (``remember_decision`` / ``decision_for``, keyed by a prompt digest)
    is emptied too: tests reuse prompt texts, and one test's remembered budget must not become
    another's "granted" budget."""
    from codna import review_budget

    monkeypatch.setenv("CODNA_REVIEW_HISTORY_DIR", str(tmp_path / "review-budget"))
    monkeypatch.setattr(review_budget, "_DECISIONS", {})


# --- Postgres queue backend (cli/codna/webhook_pg_queue.py) ------------------------------------------
# The Postgres tests need a live database. CI provides one (a `postgres` service container in the
# `cli-webhook-pg-tests` job sets CODNA_TEST_DATABASE_URL); locally, point the variable at any
# scratch database. Without it they SKIP with this message -- the only skip the webhook suite has.
PG_URL_ENV = "CODNA_TEST_DATABASE_URL"


@pytest.fixture
def pg_url():
    import os

    url = os.environ.get(PG_URL_ENV)
    if not url:
        pytest.skip(f"{PG_URL_ENV} not set: the Postgres queue tests need a live database (CI's pg job provides one)")
    pytest.importorskip("psycopg", reason="the `webhook` extra (psycopg) is not installed")
    return url


@pytest.fixture
def pg_schema(pg_url):
    """A schema of this test's own, migrated, dropped afterwards -- tests never share rows."""
    import uuid

    from codna.webhook_pg_schema import connect, drop_schema, migrate

    name = "t_" + uuid.uuid4().hex[:12]
    with connect(pg_url) as conn:
        migrate(conn, schema=name, log=lambda _m: None)
    yield name
    with connect(pg_url) as conn:
        drop_schema(conn, schema=name)


@pytest.fixture
def make_pg_queue(pg_url, pg_schema):
    """A factory for PostgresQueue instances on this test's schema; every instance is closed at
    teardown. Heartbeats are off by default so tests drive leases by hand (heartbeat_s=0)."""
    from codna.webhook_pg_queue import PostgresQueue

    made = []

    def _make(**kwargs):
        kwargs.setdefault("heartbeat_s", 0)
        kwargs.setdefault("pool_max", 4)
        queue = PostgresQueue(pg_url, schema=pg_schema, **kwargs)
        made.append(queue)
        return queue

    yield _make
    for queue in made:
        queue.close()
