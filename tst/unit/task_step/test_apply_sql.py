"""apply_sql: registration, (de)serialization, the script it builds, the SQL it refuses, and how it parses what
psql prints. The SQL itself runs against a real PostgreSQL in
tst/integration/task_step/test_service_db_sql_postgres.py."""
import json

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import apply_sql as apply_sql_module
from agent_env.task_step.task_steps.apply_sql import (
    SUMMARY_MARKER,
    ApplySqlTaskStep,
    RefusedSqlError,
    build_script,
    parse_summary,
    refusals,
    sql_relative_error,
    volatile_functions,
)
from agent_env.task_step.task_steps.service_db_sql import ServiceDbSqlError, _psql_error


def test_steps_registered_and_roundtrip():
    reg = get_task_step_registry()
    assert reg.get("apply_sql") is ApplySqlTaskStep

    s = ApplySqlTaskStep.from_dict(ApplySqlTaskStep(id="a", version=1, env_id="e", sql="SELECT 1;").to_dict())
    assert (s.type, s.env_id, s.sql, s.fail_task_on_error) == ("apply_sql", "e", "SELECT 1;", True)



def test_apply_sql_null_sql_is_blank():
    assert ApplySqlTaskStep.from_dict({"id": "a", "env_id": "e", "sql": None}).sql == ""
    assert ApplySqlTaskStep.from_dict({"id": "a", "env_id": "e"}).sql == ""


@pytest.mark.parametrize("bad", [
    lambda: ApplySqlTaskStep(id="a", version=1, env_id=""),
    lambda: ApplySqlTaskStep.from_dict({"id": "a"}),
    lambda: ApplySqlTaskStep(id="a", version=1, env_id="e", sql=["SELECT 1"]),
])
def test_config_errors(bad):
    with pytest.raises(ValueError):
        bad()


def test_volatile_functions():
    sql = ("INSERT INTO s.t VALUES (1, NOW(), random()); INSERT INTO s.t VALUES (2, now(), current_timestamp);"
           " SELECT gen_random_uuid ();")
    assert volatile_functions(sql) == ["now", "random", "current_timestamp", "gen_random_uuid"]
    assert volatile_functions("INSERT INTO s.t (created_at) VALUES ('2026-02-14T09:00:00Z');") == []
    # A column named like a function is not a call.
    assert volatile_functions("UPDATE s.t SET random_seed = 3;") == []


def test_build_script_wraps_sql_between_mark_and_cleanup():
    script = build_script("INSERT INTO s.t VALUES (1)")  # no trailing semicolon
    pre, _, rest = script.partition("INSERT INTO s.t VALUES (1)\n;\n")
    assert "CREATE TEMP TABLE _agentenv_apply_sql_mark" in pre
    assert rest.index(SUMMARY_MARKER) < rest.index("DELETE FROM public._changelog") < rest.index("setval")


def test_sql_relative_error_counts_lines_of_the_pasted_sql():
    assert sql_relative_error('psql:<stdin>:3: ERROR:  x') == "line 2 of the SQL: ERROR:  x"
    # A failure in the preamble keeps psql's own position.
    assert sql_relative_error("psql:<stdin>:1: ERROR:  y") == "psql:<stdin>:1: ERROR:  y"


def test_parse_summary():
    changes = [{"schema_name": "slack", "table_name": "messages", "operation": "INSERT", "rows": 2}]
    out = f"some user select output\n{SUMMARY_MARKER}\n{json.dumps(changes)}\n"
    assert parse_summary(out) == changes
    assert parse_summary("no marker") == []
    assert parse_summary(f"{SUMMARY_MARKER}\nnot json\n") == []


def test_psql_error_keeps_error_lines_and_names_a_missing_service():
    err = 'psql:<stdin>:4: ERROR:  relation "slack.nope" does not exist\nLINE 1: INSERT INTO slack.nope\n'
    assert _psql_error(err, "") == err.strip()
    assert "no running 'servicedb'" in _psql_error("service \"servicedb\" is not running", "")


@pytest.mark.parametrize("sql", [
    "INSERT INTO s.t VALUES (1);\nUPDATE s.t SET a = 'COMMIT; BEGIN' WHERE id = 1;",
    "DO $$ BEGIN PERFORM 1; END $$;",
    "CREATE FUNCTION f() RETURNS int AS $body$ BEGIN RETURN 1; END $body$ LANGUAGE plpgsql;",
    "-- COMMIT later\n/* ROLLBACK */ INSERT INTO s.t VALUES (E'it''s back\\tafter');",
    "INSERT INTO s.t VALUES (E'it\\'s\\tback; COMMIT');",
    "/* outer /* inner */ COMMIT; still a comment */ INSERT INTO s.t VALUES (1);",
    "INSERT INTO s.t VALUES ($tag$ COMMIT; \\! $notag$ $tag$);",
])
def test_refusals_ignore_literals_comments_and_bodies(sql):
    assert refusals(sql) == []


@pytest.mark.parametrize(("sql", "found"), [
    ("BEGIN;\nINSERT INTO s.t VALUES (1);\nCOMMIT;\nSELECT 1/0;", ["line 1 of the SQL: BEGIN", "line 3 of the SQL: COMMIT"]),
    ("INSERT INTO s.t VALUES (1); savepoint x;", ["line 1 of the SQL: SAVEPOINT"]),
    ("INSERT INTO s.t VALUES (1);\n\\! rm -rf /", ["line 2 of the SQL: psql meta-commands"]),
    # Without the E prefix a backslash is an ordinary character, so the quote after it ends the string.
    ("INSERT INTO s.t VALUES ('a\\');\nCOMMIT;", ["line 2 of the SQL: COMMIT"]),
    ("INSERT INTO s.t VALUES (E'a\\'');\nCOMMIT;", ["line 2 of the SQL: COMMIT"]),
])
def test_refusals_name_transaction_control_and_meta_commands(sql, found):
    problems = refusals(sql)
    assert len(problems) == len(found)
    assert all(p.startswith(f) for p, f in zip(problems, found))


class _FakeHost:
    def __init__(self, fail: bool = False):
        self.scripts, self.fail = [], fail

    async def run_sql_file(self, what, sql, *flags):
        self.scripts.append((sql, flags))
        if self.fail:
            raise ServiceDbSqlError(what, 3, 'psql:<stdin>:2: ERROR:  syntax error at or near "INSRT"')
        return f"{SUMMARY_MARKER}\n" + json.dumps(
            [{"schema_name": "s", "table_name": "t", "operation": "INSERT", "rows": 1}])


@pytest.fixture
def host(monkeypatch):
    holder = {}

    async def fake(context, env_id, what):
        return holder["host"]

    monkeypatch.setattr(apply_sql_module, "service_db_host", fake)
    return holder


@pytest.mark.asyncio
async def test_apply_sql_execute_records_changes(host):
    host["host"] = h = _FakeHost()
    ctx = TaskStepContext()
    await ApplySqlTaskStep(id="inject", version=1, env_id="e", sql="INSERT INTO s.t VALUES (1, now());").execute(ctx)
    rec = ctx.metadata["applied_sql"]["inject"]
    assert rec["applied"] is True and rec["env_id"] == "e" and rec["volatile_functions"] == ["now"]
    assert rec["changes"] == [{"schema_name": "s", "table_name": "t", "operation": "INSERT", "rows": 1}]
    assert len(rec["sql_sha256"]) == 64
    (script, flags), = h.scripts
    assert "-1" in flags and "INSERT INTO s.t VALUES (1, now());" in script


@pytest.mark.asyncio
async def test_apply_sql_blank_is_a_noop_without_touching_the_env(host):
    host["host"] = h = _FakeHost()
    ctx = TaskStepContext()
    await ApplySqlTaskStep(id="inject", version=1, env_id="e", sql="  \n").execute(ctx)
    assert h.scripts == []
    assert ctx.metadata["applied_sql"]["inject"]["applied"] is False


@pytest.mark.asyncio
async def test_apply_sql_failure_raises_psql_error(host):
    host["host"] = _FakeHost(fail=True)
    ctx = TaskStepContext()
    with pytest.raises(ServiceDbSqlError, match="syntax error at or near"):
        await ApplySqlTaskStep(id="inject", version=1, env_id="e", sql="INSRT INTO s.t VALUES (1);").execute(ctx)
    assert ctx.metadata["applied_sql"]["inject"]["applied"] is False


@pytest.mark.asyncio
async def test_apply_sql_refuses_a_commit_before_running_anything(host):
    host["host"] = h = _FakeHost()
    ctx = TaskStepContext()
    with pytest.raises(RefusedSqlError, match="line 2 of the SQL: COMMIT"):
        await ApplySqlTaskStep(id="inject", version=1, env_id="e",
                               sql="INSERT INTO s.t VALUES (1);\nCOMMIT;\nSELECT 1/0;").execute(ctx)
    assert h.scripts == []
    assert ctx.metadata["applied_sql"]["inject"]["applied"] is False
