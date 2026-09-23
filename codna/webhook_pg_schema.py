"""Schema for the Postgres job queue behind the Codna GitHub App webhook, and the one command that
applies it (``codna webhook migrate``).

Numbered, ADDITIVE steps: a step only ever creates or adds, never drops or renames, so version N
code keeps running against schema N+1 and a one-release rollback needs no down-migration. The
release command runs :func:`migrate` under a Postgres advisory lock, so two machines deploying at
once cannot both apply a step. Boot refuses when the schema is OLDER than the code requires (the
code would read columns that do not exist) and only warns when it is newer by more than one step.

The dependency (``psycopg`` 3 + ``psycopg_pool``) is the ``webhook`` extra of the CLI package; a
plain ``pip install codna`` does not carry it, and nothing here is imported unless the
``postgres``/``shadow`` backend is selected (see ``webhook_backend``).
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Callable

DEFAULT_SCHEMA = "codna_webhook"
_SCHEMA_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

STATUSES = ("queued", "running", "done", "failed", "superseded", "cancelled", "dead", "exported")
KINDS = ("review", "queue", "secure", "fix")
# CLASS: 0 gates a merge (review, and the merge-group job inheriting its verdict), then secure,
# then fix. The claim ORDER BY starts here; the aging term in webhook_pg_queue lets a long-waiting
# fix compete at a better class so strict priority can never starve it.
CLASS_SQL = "CASE kind WHEN 'review' THEN 0 WHEN 'queue' THEN 1 WHEN 'secure' THEN 2 ELSE 3 END"


def _steps(schema: str) -> list[str]:
    """Every migration step, in order. ``schema_version`` records the highest applied one."""
    s = schema
    return [
        # -- step 1: the whole queue ---------------------------------------------------------
        f"""
        CREATE SCHEMA IF NOT EXISTS {s};
        CREATE TABLE IF NOT EXISTS {s}.jobs (
            id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            delivery_id       TEXT NOT NULL UNIQUE,
            idempotency_key   TEXT NOT NULL,
            kind              TEXT NOT NULL CHECK (kind IN ('review','queue','secure','fix')),
            class             SMALLINT GENERATED ALWAYS AS ({CLASS_SQL}) STORED,
            repo              TEXT NOT NULL,
            ref               TEXT,
            installation_id   BIGINT,
            issue_number      INT,
            pr_number         INT,
            context           JSONB,
            reason            TEXT NOT NULL DEFAULT '',
            status            TEXT NOT NULL DEFAULT 'queued'
                              CHECK (status IN ('queued','running','done','failed','superseded','cancelled','dead','exported')),
            priority          SMALLINT NOT NULL DEFAULT 0,
            attempts          INT NOT NULL DEFAULT 0,
            max_attempts      INT NOT NULL,
            resumes           SMALLINT NOT NULL DEFAULT 0,
            not_before        TIMESTAMPTZ,
            owner             TEXT,
            lease_expires_at  TIMESTAMPTZ,
            check_run_id      BIGINT,
            check_started_at  TIMESTAMPTZ,
            posted            JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            cancel_requested  TEXT,
            last_error        TEXT,
            result            JSONB,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            claimed_at        TIMESTAMPTZ,
            finished_at       TIMESTAMPTZ,
            src_sqlite_id     INT
        );
        CREATE INDEX IF NOT EXISTS jobs_claim_idx  ON {s}.jobs (class, priority DESC, id) WHERE status = 'queued';
        CREATE INDEX IF NOT EXISTS jobs_lease_idx  ON {s}.jobs (lease_expires_at) WHERE status = 'running';
        CREATE INDEX IF NOT EXISTS jobs_tenant_idx ON {s}.jobs (installation_id, status);
        CREATE INDEX IF NOT EXISTS jobs_finished_idx ON {s}.jobs (finished_at) WHERE finished_at IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_live_head ON {s}.jobs (repo, kind, pr_number, ref)
            WHERE status IN ('queued','running') AND kind IN ('review','fix') AND ref IS NOT NULL;
        CREATE TABLE IF NOT EXISTS {s}.tenants (
            installation_id        BIGINT PRIMARY KEY,
            paused                 BOOLEAN NOT NULL DEFAULT false,
            max_concurrency        SMALLINT NOT NULL DEFAULT 3,
            weight                 SMALLINT NOT NULL DEFAULT 1,
            dead_letter_conclusion TEXT CHECK (dead_letter_conclusion IN ('neutral','action_required')),
            updated_at             TIMESTAMPTZ,
            updated_by             TEXT
        );
        CREATE TABLE IF NOT EXISTS {s}.workers (
            owner        TEXT PRIMARY KEY,
            machine_id   TEXT,
            region       TEXT,
            image        TEXT,
            role         TEXT,
            slots        SMALLINT NOT NULL DEFAULT 0,
            busy         SMALLINT NOT NULL DEFAULT 0,
            draining     BOOLEAN NOT NULL DEFAULT false,
            drain_acked_at TIMESTAMPTZ,
            started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS {s}.job_events (
            id      BIGSERIAL PRIMARY KEY,
            job_id  BIGINT,
            at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            event   TEXT NOT NULL,
            actor   TEXT,
            detail  JSONB
        );
        CREATE INDEX IF NOT EXISTS job_events_job_idx ON {s}.job_events (job_id, id);
        CREATE TABLE IF NOT EXISTS {s}.schema_version (
            version    INT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ]


REQUIRED_VERSION = len(_steps(DEFAULT_SCHEMA))


def validate_schema_name(schema: str) -> str:
    """The schema name is interpolated into DDL, so it is restricted to a plain identifier."""
    if not _SCHEMA_NAME_RE.match(schema or ""):
        raise ValueError(f"invalid Postgres schema name {schema!r}")
    return schema


def lock_key(schema: str, purpose: str) -> str:
    """The text hashed into an advisory-lock key: per schema, so two queues in one database (the
    test suite gives every test its own schema) never contend on each other's locks."""
    return f"{validate_schema_name(schema)}:{purpose}"


def database_url(environ: Any = None) -> str | None:
    """``DATABASE_URL`` (what ``fly mpg attach`` sets) or the explicit ``CODNA_WEBHOOK_DATABASE_URL``."""
    source = os.environ if environ is None else environ
    return source.get("CODNA_WEBHOOK_DATABASE_URL") or source.get("DATABASE_URL") or None


def connect(url: str, **kwargs: Any):
    """A plain autocommit connection (psycopg 3). Import is deferred so the module loads without
    the ``webhook`` extra installed."""
    import psycopg

    kwargs.setdefault("autocommit", True)
    kwargs.setdefault("connect_timeout", 5)
    return psycopg.connect(url, **kwargs)


def current_version(conn: Any, *, schema: str = DEFAULT_SCHEMA) -> int:
    """The highest applied step, 0 when the schema has never been created."""
    s = validate_schema_name(schema)
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = 'schema_version'",
        (s,),
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute(f"SELECT COALESCE(MAX(version), 0) AS v FROM {s}.schema_version").fetchone()
    return int(_first(row)) if row else 0


def _first(row: Any) -> Any:
    """The first column of a row whether the connection yields tuples or dicts (the queue's pool
    uses ``dict_row``; the migrate CLI uses a plain connection)."""
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def migrate(conn: Any, *, schema: str = DEFAULT_SCHEMA, log: Callable[[str], None] | None = None) -> int:
    """Apply every step above the current version, one transaction each, under an advisory lock.
    Idempotent: a second run applies nothing. Returns the resulting version."""
    s = validate_schema_name(schema)
    say = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    steps = _steps(s)
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key(s, "migrate"),))
        have = current_version(conn, schema=s)
        for number, sql in enumerate(steps, start=1):
            if number <= have:
                continue
            conn.execute(sql)
            conn.execute(f"INSERT INTO {s}.schema_version (version) VALUES (%s) ON CONFLICT DO NOTHING", (number,))
            say(f"codna webhook migrate: applied schema step {number} to {s}")
    return current_version(conn, schema=s)


def check_compatible(conn: Any, *, schema: str = DEFAULT_SCHEMA) -> int:
    """Boot-time guard. Raises when the schema is older than this code needs; warns (stderr) when
    it is more than one step ahead, which means a rollback has crossed the one-release window."""
    have = current_version(conn, schema=schema)
    if have < REQUIRED_VERSION:
        raise RuntimeError(
            f"codna webhook: Postgres schema {schema} is at version {have}, this build needs "
            f"{REQUIRED_VERSION}; run `codna webhook migrate` (the Fly release command does) first."
        )
    if have > REQUIRED_VERSION + 1:
        print(f"codna webhook: schema {schema} is at version {have}, more than one step ahead of this build "
              f"({REQUIRED_VERSION}); additive steps keep it readable, but roll forward soon.",
              file=sys.stderr, flush=True)
    return have


def drop_schema(conn: Any, *, schema: str) -> None:
    """Test hygiene only: remove a per-test schema. Never called by the service."""
    s = validate_schema_name(schema)
    if s == DEFAULT_SCHEMA:
        raise ValueError("refusing to drop the production schema")
    conn.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
