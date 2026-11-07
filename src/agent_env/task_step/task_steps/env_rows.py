"""Row-level capture and diff of a deployed env's service database.

``capture_env_rows`` writes every row of every service table to a file on the env's VM host, one line per row:
``<schema>.<table> TAB <key> TAB <md5 of the row> TAB <the row as JSON>``. The key is the row's primary key as a JSON
array, or ``row:<md5>`` for a table without one. ``ignore_columns`` names columns left out of every row (and its md5),
e.g. timestamps a rebuild can't reproduce.

``diff_env_rows`` compares two captures of the same env (say: a base universe plus declared SQL, and the universe a run
actually started from), row by row, never by SQL text, and records ``context.metadata["verifications"][<step id>]``:

    {"type": "env_rows_diff", "match": bool, "expected": <capture>, "actual": <capture>,
     "counts": {"extra": n, "missing": n, "changed": n},
     "by_table": {"<schema>.<table>": {"extra": n, "missing": n, "changed": n}},
     "examples": {"extra": [{"table", "key", "row"}], "missing": [...], "changed": [{"table", "key", "expected",
                  "actual"}]}}

``extra`` rows are in ``actual`` only, ``missing`` rows in ``expected`` only, ``changed`` rows share a key but differ.
With ``fail_on_mismatch`` a mismatch also fails the step.

Typical task: ``deploy_env -> load_artifact(base) -> apply_sql -> capture_env_rows(expected) ->
load_artifact(run universe) -> capture_env_rows(actual) -> diff_env_rows``.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
from collections import defaultdict
from typing import Optional

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.service_db_sql import WORK_DIR, ServiceDbHost, service_db_host

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Emits one COPY per service table and runs each (\gexec); every COPY writes that table's rows as capture lines.
_CAPTURE_SQL = """\
SELECT format(
  'COPY (SELECT %L, %s, md5((to_jsonb(t) - %L::text[])::text), (to_jsonb(t) - %L::text[])::text FROM %I.%I t) TO STDOUT',
  n.nspname || '.' || c.relname,
  coalesce(
    (SELECT 'jsonb_build_array(' || string_agg(format('t.%I', a.attname), ', ' ORDER BY k.ord) || ')::text'
     FROM pg_index i
     CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
     JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
     WHERE i.indrelid = c.oid AND i.indisprimary),
    format('''row:'' || md5((to_jsonb(t) - %L::text[])::text)', :'ignore')),
  :'ignore', :'ignore', n.nspname, c.relname)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p')
  AND NOT c.relispartition
  AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'agent_env_internal')
  AND n.nspname NOT LIKE 'pg\\_%'
  AND NOT (n.nspname = 'public' AND c.relname = '_changelog')
ORDER BY n.nspname, c.relname
\\gexec
"""


def _check_name(value: str, field: str, step: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ValueError(f"{step}: `{field}` must be 1-64 letters, digits, '_' or '-'; got {value!r}")
    return value


def capture_path(name: str) -> str:
    return f"{WORK_DIR}/captures/{name}.tsv"


def pg_text_array(values: list[str]) -> str:
    """``values`` as a PostgreSQL text-array literal, e.g. ``{"a","b"}``."""
    return "{" + ",".join('"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' for v in values) + "}"


def parse_capture_index(text: str) -> dict[str, dict[str, list[str]]]:
    """``<table> TAB <key> TAB <md5>`` lines as {table: {key: [md5, ...]}} (a key repeats in a table without a primary
    key)."""
    index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        index[parts[0]][parts[1]].append(parts[2])
    return index


def diff_indexes(expected: dict, actual: dict) -> dict[str, dict[str, list[str]]]:
    """{"extra" | "missing" | "changed": [(table, key), ...]} between two capture indexes, each list sorted."""
    out: dict[str, list] = {"extra": [], "missing": [], "changed": []}
    for table in sorted(set(expected) | set(actual)):
        exp, act = expected.get(table, {}), actual.get(table, {})
        for key in sorted(set(exp) | set(act)):
            e, a = sorted(exp.get(key, [])), sorted(act.get(key, []))
            if e == a:
                continue
            if not e:
                out["extra"].extend([(table, key)] * len(a))
            elif not a:
                out["missing"].extend([(table, key)] * len(e))
            elif len(e) == len(a):
                out["changed"].append((table, key))
            else:  # same key, a different number of copies: a table without a primary key
                kind = "extra" if len(a) > len(e) else "missing"
                out[kind].extend([(table, key)] * abs(len(a) - len(e)))
    return out


def _unescape_copy(value: str) -> str:
    """Undo COPY text-format escaping (``\\t``, ``\\n``, ``\\\\``, ...) in one field."""
    return re.sub(r"\\(.)", lambda m: {"t": "\t", "n": "\n", "r": "\r", "b": "\b", "f": "\f", "v": "\v"}.get(
        m.group(1), m.group(1)), value)


def _row(text: Optional[str]):
    if text is None:
        return None
    raw = _unescape_copy(text)
    try:
        return json.loads(raw)
    except ValueError:
        return raw


class CaptureEnvRowsTaskStep(TaskStep):
    """Capture every row of a deployed env's service tables to a named file on its VM host."""

    type = "capture_env_rows"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, capture: str,
                 ignore_columns: Optional[list[str]] = None, depends_on: Optional[list] = None,
                 fail_task_on_error: bool = True):
        if not env_id:
            raise ValueError("capture_env_rows requires an env_id")
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.capture = _check_name(capture, "capture", "capture_env_rows")
        self.ignore_columns = [str(c) for c in (ignore_columns or [])]

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(env_id=self.env_id, capture=self.capture, ignore_columns=self.ignore_columns)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "CaptureEnvRowsTaskStep":
        if not data.get("env_id"):
            raise ValueError("capture_env_rows step requires 'env_id'")
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], capture=data.get("capture", ""),
                   ignore_columns=data.get("ignore_columns"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        host = await service_db_host(context, self.env_id, "capture_env_rows")
        path = capture_path(self.capture)
        sql_path = f"{WORK_DIR}/capture-{self.capture}.sql"
        await host.write(sql_path, _CAPTURE_SQL)
        ignore = pg_text_array(self.ignore_columns)
        cmd = (f"mkdir -p {shlex.quote(WORK_DIR)}/captures && "
               f"{host.psql('-A', '-t', '-v', shlex.quote('ignore=' + ignore), '-f', '-')} "
               f"< {shlex.quote(sql_path)} > {shlex.quote(path)}")
        code, _, err = await host.run(cmd)
        await host.run(f"rm -f {shlex.quote(sql_path)}")
        if code != 0:
            from agent_env.task_step.task_steps.service_db_sql import ServiceDbSqlError, _psql_error

            raise ServiceDbSqlError(f"capture_env_rows '{self.id}' on env '{self.env_id}'", code, _psql_error(err, ""))
        code, counts, err = await host.run(f"cut -f1 {shlex.quote(path)} | sort | uniq -c")
        tables = {}
        for line in counts.splitlines():
            n, _, table = line.strip().partition(" ")
            if table:
                tables[table] = int(n)
        context.metadata.setdefault("env_row_captures", {})[self.capture] = {
            "env_id": self.env_id, "path": path, "rows": sum(tables.values()), "tables": tables,
            "ignore_columns": self.ignore_columns,
        }
        logger.info("capture_env_rows '%s': %d rows in %d tables of env '%s' -> %s", self.id, sum(tables.values()),
                    len(tables), self.env_id, path)
        return context


class DiffEnvRowsTaskStep(TaskStep):
    """Diff two row captures of an env; record extra, missing and changed rows as a verification."""

    type = "diff_env_rows"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, expected: str, actual: str,
                 max_examples: int = 50, fail_on_mismatch: bool = False, depends_on: Optional[list] = None,
                 fail_task_on_error: bool = True):
        if not env_id:
            raise ValueError("diff_env_rows requires an env_id")
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.expected = _check_name(expected, "expected", "diff_env_rows")
        self.actual = _check_name(actual, "actual", "diff_env_rows")
        if self.expected == self.actual:
            raise ValueError("diff_env_rows: `expected` and `actual` must name different captures")
        self.max_examples = max(0, int(max_examples))
        self.fail_on_mismatch = bool(fail_on_mismatch)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(env_id=self.env_id, expected=self.expected, actual=self.actual, max_examples=self.max_examples,
                    fail_on_mismatch=self.fail_on_mismatch)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "DiffEnvRowsTaskStep":
        if not data.get("env_id"):
            raise ValueError("diff_env_rows step requires 'env_id'")
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], expected=data.get("expected", ""),
                   actual=data.get("actual", ""), max_examples=data.get("max_examples", 50),
                   fail_on_mismatch=data.get("fail_on_mismatch", False))

    async def _index(self, host: ServiceDbHost, name: str) -> dict:
        path = capture_path(name)
        code, out, err = await host.run(f"test -f {shlex.quote(path)} && cut -f1-3 {shlex.quote(path)}")
        if code != 0:
            raise RuntimeError(f"diff_env_rows '{self.id}': no capture '{name}' on env '{self.env_id}'; run a "
                               f"capture_env_rows step with capture='{name}' first ({err.strip()[-200:]})")
        return parse_capture_index(out)

    async def _rows(self, host: ServiceDbHost, name: str, keys: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """The row JSON of each (table, key) in capture ``name``."""
        if not keys:
            return {}
        keys_path = f"{WORK_DIR}/diff-{self.id}-{name}.keys"
        await host.write(keys_path, "".join(f"{t}\t{k}\n" for t, k in keys))
        script = ("awk -F'\\t' 'NR==FNR{want[$0]=1; next} (($1 FS $2) in want){print $1 FS $2 FS $4}' "
                  f"{shlex.quote(keys_path)} {shlex.quote(capture_path(name))}; rm -f {shlex.quote(keys_path)}")
        _, out, _ = await host.run(script)
        rows: dict[tuple[str, str], str] = {}
        for line in out.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                rows.setdefault((parts[0], parts[1]), parts[2])
        return rows

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        host = await service_db_host(context, self.env_id, "diff_env_rows")
        expected, actual = await self._index(host, self.expected), await self._index(host, self.actual)
        diff = diff_indexes(expected, actual)
        by_table: dict[str, dict[str, int]] = {}
        for kind, items in diff.items():
            for table, _ in items:
                by_table.setdefault(table, {"extra": 0, "missing": 0, "changed": 0})[kind] += 1
        n = self.max_examples
        extra, missing, changed = diff["extra"][:n], diff["missing"][:n], diff["changed"][:n]
        exp_rows = await self._rows(host, self.expected, sorted(set(missing + changed)))
        act_rows = await self._rows(host, self.actual, sorted(set(extra + changed)))
        result = {
            "type": "env_rows_diff",
            "match": not any(diff.values()),
            "expected": self.expected,
            "actual": self.actual,
            "counts": {k: len(v) for k, v in diff.items()},
            "by_table": by_table,
            "examples": {
                "extra": [{"table": t, "key": k, "row": _row(act_rows.get((t, k)))} for t, k in extra],
                "missing": [{"table": t, "key": k, "row": _row(exp_rows.get((t, k)))} for t, k in missing],
                "changed": [{"table": t, "key": k, "expected": _row(exp_rows.get((t, k))),
                             "actual": _row(act_rows.get((t, k)))} for t, k in changed],
            },
        }
        context.metadata.setdefault("verifications", {})[self.id] = result
        summary = ", ".join(f"{v} {k}" for k, v in result["counts"].items())
        if result["match"]:
            logger.info("diff_env_rows '%s': '%s' and '%s' match row for row", self.id, self.expected, self.actual)
        else:
            logger.warning("diff_env_rows '%s': '%s' vs '%s' differ: %s (tables: %s)", self.id, self.expected,
                           self.actual, summary, ", ".join(sorted(by_table)))
            if self.fail_on_mismatch:
                raise RuntimeError(f"diff_env_rows '{self.id}': the env's rows differ from '{self.expected}': {summary}"
                                   f" in {', '.join(sorted(by_table))}")
        return context
