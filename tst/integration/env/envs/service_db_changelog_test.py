"""Integration test: the ``public._changelog`` trigger records ``row_id`` from
each table's real PRIMARY KEY — not a hardcoded ``id`` column.

Universe tables use varied PKs (e.g. ``record_id``), so the old
``row_to_json(NEW) ->> 'id'`` extraction left ``row_id`` empty and updates could
not be tied to a record. This spins up a throwaway Postgres, applies the actual
``LocalPostgresStateProvider`` changelog init SQL, and exercises INSERT/UPDATE/DELETE on tables
whose PK is not named ``id``.
"""

import json
import subprocess
import time
import uuid

import psycopg2
import pytest

from agent_env.providers.env_state import LocalPostgresStateProvider

pytestmark = [pytest.mark.integration]

_PG_IMAGE = "postgres:16-alpine"
_PG_PASSWORD = "test"


@pytest.fixture(scope="module")
def pg_conn():
    """A ready psycopg2 connection to a throwaway Postgres container."""
    name = f"agentenv-changelog-it-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker", "run", "-d", "--rm", "--name", name,
            "-e", f"POSTGRES_PASSWORD={_PG_PASSWORD}",
            "-P", _PG_IMAGE,
        ],
        check=True, capture_output=True, text=True,
    )
    try:
        host_port = (
            subprocess.run(
                ["docker", "port", name, "5432/tcp"],
                check=True, capture_output=True, text=True,
            )
            .stdout.strip()
            .splitlines()[0]
            .rsplit(":", 1)[-1]
        )
        conn = None
        last_err = None
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                conn = psycopg2.connect(
                    host="127.0.0.1",
                    port=int(host_port),
                    user="postgres",
                    password=_PG_PASSWORD,
                    dbname="postgres",
                )
                conn.autocommit = True
                break
            except psycopg2.OperationalError as exc:
                last_err = exc
                time.sleep(1)
        if conn is None:
            raise RuntimeError(f"Postgres container never became ready: {last_err}")
        yield conn
        conn.close()
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _exec(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)


def _rows(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def test_changelog_row_id_single_column_pk(pg_conn):
    """A table whose PK is ``record_id`` (not ``id``) — row_id must be that PK."""
    schema = "health"
    # Apply the env's real changelog init SQL (schema + _changelog table + fns).
    _exec(pg_conn, LocalPostgresStateProvider.get_init_script([schema]))
    _exec(
        pg_conn,
        f'CREATE TABLE "{schema}".sleep_records (record_id text PRIMARY KEY, stage text)',
    )
    # Base row is loaded BEFORE triggers exist (mirrors load-universe) → not logged.
    _exec(pg_conn, f"INSERT INTO \"{schema}\".sleep_records VALUES ('4001', 'inBed')")
    _exec(pg_conn, f"SELECT public._install_changelog_triggers('{schema}')")

    _exec(
        pg_conn,
        f"UPDATE \"{schema}\".sleep_records SET stage='awake' WHERE record_id='4001'",
    )
    _exec(pg_conn, f"INSERT INTO \"{schema}\".sleep_records VALUES ('9999', 'asleep')")
    _exec(pg_conn, f"DELETE FROM \"{schema}\".sleep_records WHERE record_id='4001'")

    rows = _rows(
        pg_conn,
        "SELECT operation, row_id FROM public._changelog "
        f"WHERE schema_name='{schema}' AND table_name='sleep_records' ORDER BY id",
    )
    assert rows == [("UPDATE", "4001"), ("INSERT", "9999"), ("DELETE", "4001")]


def test_changelog_row_id_composite_pk(pg_conn):
    """Composite PK → row_id is an unambiguous JSON object keyed by column."""
    schema = "metrics"
    _exec(pg_conn, LocalPostgresStateProvider.get_init_script([schema]))
    _exec(
        pg_conn,
        f'CREATE TABLE "{schema}".daily '
        "(user_id text, day text, steps int, PRIMARY KEY (user_id, day))",
    )
    _exec(pg_conn, f"SELECT public._install_changelog_triggers('{schema}')")
    _exec(pg_conn, f"INSERT INTO \"{schema}\".daily VALUES ('u_jin', '2025-11-05', 100)")

    row_id = _rows(
        pg_conn,
        "SELECT row_id FROM public._changelog "
        f"WHERE schema_name='{schema}' AND table_name='daily' ORDER BY id DESC LIMIT 1",
    )[0][0]
    assert json.loads(row_id) == {"user_id": "u_jin", "day": "2025-11-05"}


def test_internal_tables_get_no_changelog_trigger(pg_conn):
    """A service's internal tables (a leading underscore, like _changelog) hold no
    domain rows: the installer skips them, so a blob written there is not copied
    into changed_fields while the domain table beside it is still audited."""
    schema = "files_svc"
    _exec(pg_conn, LocalPostgresStateProvider.get_init_script([schema]))
    _exec(pg_conn, f'CREATE TABLE "{schema}".documents (id text PRIMARY KEY, title text)')
    _exec(pg_conn, f'CREATE TABLE "{schema}"._files (path text PRIMARY KEY, content bytea)')
    _exec(pg_conn, f"SELECT public._install_changelog_triggers('{schema}')")

    _exec(pg_conn, f"INSERT INTO \"{schema}\".documents VALUES ('d1', 'brief')")
    _exec(pg_conn, f"INSERT INTO \"{schema}\"._files VALUES ('a/brief.pdf', '\\x255044462d')")

    rows = _rows(
        pg_conn,
        "SELECT table_name, operation FROM public._changelog "
        f"WHERE schema_name='{schema}' ORDER BY id",
    )
    assert rows == [("documents", "INSERT")]
    triggers = _rows(
        pg_conn,
        "SELECT event_object_table FROM information_schema.triggers "
        f"WHERE trigger_schema='{schema}' AND trigger_name='_changelog_trigger'",
    )
    assert {t[0] for t in triggers} == {"documents"}
