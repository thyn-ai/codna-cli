"""Make the `codna` package and shared test helpers importable from any pytest invocation.

`codna.sarif` / `codna.findings` are pure-stdlib, so importing the package does not pull
in the engine client — the offline SARIF suite runs without installing the full CLI.
"""
import contextlib
import pathlib
import sys

import pytest

CLI_DIR = pathlib.Path(__file__).resolve().parents[1]  # .../cli
TESTS_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
for import_dir in (CLI_DIR, TESTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))


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


@contextlib.contextmanager
def unlistable_directory(path):
    """``path`` made unlistable, or a SKIP when this machine cannot make it so.

    WHY THIS IS A HELPER AND A SKIP RATHER THAN A BARE ``chmod(0o000)`` IN EACH TEST. Two tests need a real
    unreadable directory: the skill-discovery warning and the bridge that carries it to the host. A bare
    ``chmod`` is a silent assumption with two failure modes, and BOTH make a test dishonest:

      * running as **root** ignores the mode entirely (``CAP_DAC_OVERRIDE``), so nothing raises and an assertion
        on the warning fails for a reason that has nothing to do with the code;
      * a filesystem that does not carry POSIX mode bits would do the same.

    So the mode is APPLIED and then PROVEN by attempting the very operation the code under test performs
    (``os.listdir``). When it does not raise, there is nothing to test and the callers skip with the reason
    instead of asserting on a precondition that never held. The yielded path is always restored, so a failing
    assertion cannot leave an unlistable directory behind.
    """
    import errno
    import os

    original = os.stat(path).st_mode
    os.chmod(path, 0o000)
    try:
        try:
            os.listdir(path)
        except OSError as denied:
            if denied.errno not in (errno.EACCES, errno.EPERM):
                raise
            yield path
        else:
            reason = (
                "running as root" if hasattr(os, "geteuid") and os.geteuid() == 0
                else "the filesystem does not enforce POSIX mode bits"
            )
            pytest.skip(
                f"cannot make a directory unlistable ({reason}); a chmod-based unreadable-directory test "
                "would assert on a precondition this environment does not provide"
            )
    finally:
        os.chmod(path, original)


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
