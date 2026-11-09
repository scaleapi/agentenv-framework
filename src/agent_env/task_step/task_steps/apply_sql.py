"""Apply SQL to a deployed env's service database, as part of building the state a run starts from.

Typical use: ``deploy_env -> load_artifact (a base universe) -> apply_sql -> deploy_agent -> prompt_agent``, so the
agent starts from the base universe plus exactly the declared SQL, and nothing else.

The SQL runs as one psql script in a single transaction (``psql -1``) and stops at its first error, which fails the
step with psql's message (it names the statement's line). After it:

- the env's changelog rows the SQL wrote are deleted, so the run starts with a changelog that holds only what
  happens after this step, as a freshly loaded universe does; nothing the SQL did shows up as an agent's change;
- every serial / identity sequence in the services' schemas is advanced past its column's largest value, so an
  insert with explicit ids doesn't make the services' own later inserts collide;
- SQL that would end or leave that transaction (``BEGIN``, ``COMMIT``, ``ROLLBACK``, ``SAVEPOINT``, ... at the start
  of a statement) or holds a psql meta-command (a backslash outside a literal) is refused before anything runs, since
  a ``COMMIT`` would keep the writes before a later failing statement;
- ``context.metadata["applied_sql"][<step id>]`` records the SQL's sha256 and size, the rows it wrote per table and
  operation (read from the changelog before it was cleared), and any volatile functions (``now()``, ``random()``,
  ...) it calls: those give a different result on every apply, so a rebuild can't reproduce their rows exactly.

Blank SQL is a no-op that still records the sha256 of the empty string, so a task built from an empty field runs on
the base universe alone.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Optional

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.service_db_sql import ServiceDbHost, ServiceDbSqlError, service_db_host

logger = logging.getLogger(__name__)

SUMMARY_MARKER = "__agentenv_apply_sql_summary__"

# Calls whose result differs on every apply.
_VOLATILE = re.compile(
    r"\b(now|clock_timestamp|statement_timestamp|transaction_timestamp|timeofday|random|gen_random_uuid|"
    r"uuid_generate_v[14])\s*\(|\b(current_timestamp|current_date|current_time|localtimestamp|localtime)\b",
    re.IGNORECASE,
)

# Before the SQL: remember where the changelog ends.
_PREAMBLE = """\
CREATE TEMP TABLE _agentenv_apply_sql_mark AS SELECT coalesce(max(id), 0) AS pre FROM public._changelog;
"""

# After the SQL: summarize and clear its changelog rows, then advance every service sequence past its column's max.
_EPILOGUE = f"""\
;
\\echo {SUMMARY_MARKER}
SELECT coalesce(jsonb_agg(s ORDER BY s.schema_name, s.table_name, s.operation), '[]'::jsonb)
FROM (SELECT schema_name, table_name, operation, count(*) AS rows FROM public._changelog
      WHERE id > (SELECT pre FROM _agentenv_apply_sql_mark) GROUP BY 1, 2, 3) s;
DELETE FROM public._changelog WHERE id > (SELECT pre FROM _agentenv_apply_sql_mark);
DO $agentenv_sync$
DECLARE r record; mx bigint; cur bigint; called boolean;
BEGIN
  FOR r IN
    SELECT n.nspname AS sch, c.relname AS tbl, a.attname AS col,
           pg_get_serial_sequence(format('%I.%I', n.nspname, c.relname), a.attname) AS seq
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'agent_env_internal')
      AND n.nspname NOT LIKE 'pg\\_%'
      AND pg_get_serial_sequence(format('%I.%I', n.nspname, c.relname), a.attname) IS NOT NULL
  LOOP
    EXECUTE format('SELECT max(%I)::bigint FROM %I.%I', r.col, r.sch, r.tbl) INTO mx;
    EXECUTE format('SELECT last_value, is_called FROM %s', r.seq) INTO cur, called;
    IF mx IS NOT NULL AND (mx > cur OR (mx = cur AND NOT called)) THEN
      PERFORM setval(r.seq, mx, true);
    END IF;
  END LOOP;
END
$agentenv_sync$;
"""


# Comments, string literals, quoted identifiers and dollar-quoted bodies, which hold no statement of the script's own.
_LITERAL = re.compile(
    r"--[^\n]*|/\*.*?\*/|'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|\$([A-Za-z_][A-Za-z0-9_]*|)\$.*?\$\1\$",
    re.DOTALL,
)
_TRANSACTION_CONTROL = re.compile(
    r"(?:\A|;)\s*(begin|commit|rollback|end|abort|savepoint|release|start\s+transaction|prepare\s+transaction)\b",
    re.IGNORECASE,
)


class RefusedSqlError(ServiceDbSqlError):
    """SQL ``apply_sql`` won't run, refused before it reaches the database."""

    def __init__(self, what: str, message: str):
        RuntimeError.__init__(self, f"{what} refused: {message}")
        self.exit_code = None
        self.message = message


def refusals(sql: str) -> list[str]:
    """Why ``sql`` can't run as one ``apply_sql`` transaction, each naming its line; empty when it can."""
    code = _LITERAL.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), sql)
    found = []
    for m in _TRANSACTION_CONTROL.finditer(code):
        line = code.count("\n", 0, m.start(1)) + 1
        found.append(f"line {line} of the SQL: {m.group(1).upper()} controls the transaction apply_sql runs the SQL "
                     "in; remove it (the step commits on success and rolls everything back on an error)")
    for m in re.finditer(r"\\", code):
        line = code.count("\n", 0, m.start()) + 1
        found.append(f"line {line} of the SQL: psql meta-commands (a backslash outside a literal) aren't allowed")
    return found


def volatile_functions(sql: str) -> list[str]:
    """The volatile calls ``sql`` makes, lower-cased and de-duplicated, in order of first use."""
    seen: dict[str, None] = {}
    for m in _VOLATILE.finditer(sql):
        seen.setdefault((m.group(1) or m.group(2)).lower(), None)
    return list(seen)


def build_script(sql: str) -> str:
    """The psql script ``apply_sql`` runs: the preamble, the SQL, then the epilogue."""
    return _PREAMBLE + sql.rstrip() + "\n" + _EPILOGUE


_PSQL_LINE = re.compile(r"psql:<stdin>:(\d+):")


def sql_relative_error(message: str) -> str:
    """psql's ``psql:<stdin>:N:`` positions, which count the preamble's lines, as lines of the SQL itself."""
    offset = _PREAMBLE.count("\n")

    def line(m: re.Match) -> str:
        n = int(m.group(1)) - offset
        return f"line {n} of the SQL:" if n >= 1 else m.group(0)

    return _PSQL_LINE.sub(line, message)


def parse_summary(stdout: str) -> list[dict]:
    """The per-table write counts the epilogue prints after its marker; [] when absent or unparsable."""
    _, found, rest = stdout.partition(SUMMARY_MARKER)
    if not found:
        return []
    for line in rest.splitlines():
        line = line.strip()
        if line.startswith("["):
            try:
                return json.loads(line)
            except ValueError:
                return []
    return []


class ApplySqlTaskStep(TaskStep):
    """Run SQL against a deployed env's service database, then clear the changelog rows it wrote."""

    type = "apply_sql"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        sql: str = "",
        depends_on: Optional[list] = None,
        fail_task_on_error: bool = True,
    ):
        if not env_id:
            raise ValueError("apply_sql requires an env_id (the env whose service database the SQL runs in)")
        if not isinstance(sql, str):
            raise ValueError("apply_sql: `sql` must be a string")
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.sql = sql

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["sql"] = self.sql
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "ApplySqlTaskStep":
        if not data.get("env_id"):
            raise ValueError("apply_sql step requires 'env_id'")
        sql = data.get("sql")
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], sql="" if sql is None else sql)

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        record = sql_record(self.sql, self.env_id)
        context.metadata.setdefault("applied_sql", {})[self.id] = record
        if not (self.sql or "").strip():
            logger.info("apply_sql '%s': no SQL; env '%s' keeps the state it was loaded with", self.id, self.env_id)
            return context
        host = await service_db_host(context, self.env_id, "apply_sql")
        await apply_sql_on_host(host, self.sql, record, what=f"apply_sql '{self.id}' on env '{self.env_id}'")
        return context


def sql_record(sql: str, env_id: str) -> dict:
    """What ``apply_sql`` records for ``sql`` before applying it; ``apply_sql_on_host`` fills in the rest."""
    sql = sql or ""
    return {
        "env_id": env_id,
        "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        "sql_bytes": len(sql.encode("utf-8")),
        "volatile_functions": volatile_functions(sql),
        "changes": [],
        "applied": False,
    }


async def apply_sql_on_host(host: ServiceDbHost, sql: str, record: dict, *, what: str) -> dict:
    """Apply ``sql`` to the service database on ``host`` as ``apply_sql`` does, filling in ``record`` (from
    ``sql_record``): the rows written per table, and ``applied``. For callers that restore an env outside a task,
    such as a trajectory editor rebuilding a run's start state. Raises ``ServiceDbSqlError`` with the failing line
    counted in ``sql`` as given, or ``RefusedSqlError`` before anything runs."""
    if problems := refusals(sql):
        raise RefusedSqlError(what, "; ".join(problems))
    if record["volatile_functions"]:
        logger.warning("%s: the SQL calls %s, which give a different result on every apply", what,
                       ", ".join(record["volatile_functions"]))
    try:
        out = await host.run_sql_file(what, build_script(sql), "-1", "-A", "-t")
    except ServiceDbSqlError as e:
        raise ServiceDbSqlError(what, e.exit_code, sql_relative_error(e.message)) from None
    record["changes"] = parse_summary(out)
    record["applied"] = True
    logger.info("%s: applied %d bytes: %s", what, record["sql_bytes"],
                ", ".join(f"{c['operation']} {c['rows']} {c['schema_name']}.{c['table_name']}"
                          for c in record["changes"]) or "no rows written")
    return record
