"""apply_sql, capture_env_rows and diff_env_rows: registration, (de)serialization, the scripts they build, and how
they parse what psql prints. The SQL itself runs against a real PostgreSQL in
tst/integration/task_step/test_service_db_sql_postgres.py."""
import json

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import apply_sql as apply_sql_module
from agent_env.task_step.task_steps.apply_sql import (
    SUMMARY_MARKER,
    ApplySqlTaskStep,
    build_script,
    parse_summary,
    sql_relative_error,
    volatile_functions,
)
from agent_env.task_step.task_steps.env_rows import (
    CaptureEnvRowsTaskStep,
    DiffEnvRowsTaskStep,
    diff_indexes,
    parse_capture_index,
    pg_text_array,
)
from agent_env.task_step.task_steps.service_db_sql import ServiceDbSqlError, _psql_error


def test_steps_registered_and_roundtrip():
    reg = get_task_step_registry()
    assert reg.get("apply_sql") is ApplySqlTaskStep
    assert reg.get("capture_env_rows") is CaptureEnvRowsTaskStep
    assert reg.get("diff_env_rows") is DiffEnvRowsTaskStep

    s = ApplySqlTaskStep.from_dict(ApplySqlTaskStep(id="a", version=1, env_id="e", sql="SELECT 1;").to_dict())
    assert (s.type, s.env_id, s.sql, s.fail_task_on_error) == ("apply_sql", "e", "SELECT 1;", True)

    c = CaptureEnvRowsTaskStep.from_dict(
        CaptureEnvRowsTaskStep(id="c", version=1, env_id="e", capture="expected", ignore_columns=["ts"]).to_dict())
    assert (c.capture, c.ignore_columns) == ("expected", ["ts"])

    d = DiffEnvRowsTaskStep.from_dict(DiffEnvRowsTaskStep(
        id="d", version=1, env_id="e", expected="expected", actual="actual", fail_on_mismatch=True).to_dict())
    assert (d.expected, d.actual, d.fail_on_mismatch, d.max_examples) == ("expected", "actual", True, 50)


def test_apply_sql_null_sql_is_blank():
    assert ApplySqlTaskStep.from_dict({"id": "a", "env_id": "e", "sql": None}).sql == ""
    assert ApplySqlTaskStep.from_dict({"id": "a", "env_id": "e"}).sql == ""


@pytest.mark.parametrize("bad", [
    lambda: ApplySqlTaskStep(id="a", version=1, env_id=""),
    lambda: ApplySqlTaskStep.from_dict({"id": "a"}),
    lambda: ApplySqlTaskStep(id="a", version=1, env_id="e", sql=["SELECT 1"]),
    lambda: CaptureEnvRowsTaskStep(id="c", version=1, env_id="e", capture="../x"),
    lambda: CaptureEnvRowsTaskStep(id="c", version=1, env_id="e", capture=""),
    lambda: DiffEnvRowsTaskStep(id="d", version=1, env_id="e", expected="a", actual="a"),
    lambda: DiffEnvRowsTaskStep(id="d", version=1, env_id="e", expected="a b", actual="c"),
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


def test_pg_text_array_quotes():
    assert pg_text_array([]) == "{}"
    assert pg_text_array(["a", 'b"c', "d\\e"]) == '{"a","b\\"c","d\\\\e"}'


def test_diff_indexes():
    expected = parse_capture_index(
        "s.t\t[1]\taaa\n"
        "s.t\t[2]\tbbb\n"
        "s.t\t[3]\tccc\n"
        "s.nopk\trow:x\tx\n"
    )
    actual = parse_capture_index(
        "s.t\t[1]\taaa\n"   # same
        "s.t\t[2]\tBBB\n"   # changed
        "s.t\t[4]\tddd\n"   # extra; [3] missing
        "s.nopk\trow:x\tx\n"
        "s.nopk\trow:x\tx\n"  # a duplicate row in a table without a primary key
        "s.new\t[1]\teee\n"  # a whole table only in actual
    )
    assert diff_indexes(expected, actual) == {
        "extra": [("s.new", "[1]"), ("s.nopk", "row:x"), ("s.t", "[4]")],
        "missing": [("s.t", "[3]")],
        "changed": [("s.t", "[2]")],
    }
    assert diff_indexes(expected, expected) == {"extra": [], "missing": [], "changed": []}


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
