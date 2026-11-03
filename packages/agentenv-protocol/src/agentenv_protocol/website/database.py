"""A site whose state is a database's: ``@website(models=...)`` names the data an MCP server keeps, so the site's pages
and the server's tools read and write the same rows under the same constraints (and the same rules, where the site
calls the server's own service code). ``models`` is either an object with
``load(data)``, ``export()`` and ``reset()`` (each may be async), the data hooks themselves, or SQLAlchemy models: a
declarative ``Base``, a ``MetaData``, or a module holding one (its ``Base``, else its one metadata), as the
synthetic_servers kit's ``database.py`` declares them. For models, the site's ``self.db`` is a SQLAlchemy session
factory on ``DATABASE_URL``, which agent-env points at the env's own schema of the service database (its
``search_path``), creating the models' tables there on first use; its data is a JSON object of each table's rows by
table name, which data/add inserts, data/get exports, and data/reset deletes. SQLAlchemy is imported only for models,
from the ``db`` extra, which carries SQLAlchemy 2.1 and its ``postgresql://`` driver, ``psycopg``; a site whose
``DATABASE_URL`` names another driver installs that one, and an engine whose driver isn't installed raises ImportError
naming it.

Beside the tables, the data's string ``current_user`` names the user a browser starts signed in as, as the default
store's does, and the state data/add last loaded is the initial one ``/go`` reports. A models site keeps one state,
the database's, so ``/go`` answers it for every ``sid``."""
from __future__ import annotations

import copy
import datetime
import decimal
import itertools
import json
import types
import uuid
from typing import Any, Callable, Iterable

from pydantic_core import to_jsonable_python

from ..agent_env_environment import _maybe_await
from .store import CURRENT_USER, part_data, verifier_state

HOOKS = ("load", "export", "reset")


def is_hooks(models: Any) -> bool:
    """Whether ``models`` is data hooks of its own: an object with callable ``load``, ``export`` and ``reset``."""
    return all(callable(getattr(models, name, None)) for name in HOOKS)


def metadata_of(models: Any) -> Any:
    """The SQLAlchemy ``MetaData`` ``models`` names: itself, its ``metadata``, or a module's ``Base``'s or one
    metadata; raises TypeError for anything else."""
    sqlalchemy = _sqlalchemy()
    if isinstance(models, sqlalchemy.MetaData):
        return models
    if isinstance(getattr(models, "metadata", None), sqlalchemy.MetaData):
        return models.metadata
    if isinstance(models, types.ModuleType):
        if isinstance(getattr(getattr(models, "Base", None), "metadata", None), sqlalchemy.MetaData):
            return models.Base.metadata
        found = {id(m): m for value in vars(models).values() if isinstance(
            m := value if isinstance(value, sqlalchemy.MetaData) else getattr(value, "metadata", None),
            sqlalchemy.MetaData)}
        if len(found) == 1:
            return next(iter(found.values()))
        if found:
            raise TypeError(f"module {models.__name__} holds {len(found)} MetaData; pass models= its Base")
    raise TypeError(f"models= is an object with load/export/reset, or SQLAlchemy models (a declarative Base, a "
                    f"MetaData, or a module holding one), not {models!r}")


def _sqlalchemy() -> Any:
    try:
        import sqlalchemy
    except ImportError as e:
        raise ImportError("@website(models=...) of SQLAlchemy models needs sqlalchemy: "
                          "pip install 'agentenv-framework-protocol[db]'") from e
    return sqlalchemy


class Tables:
    """The data hooks of SQLAlchemy ``metadata`` on the database at ``url()``: each table's rows by table name."""

    def __init__(self, metadata: Any, url: Callable[[], str | None]) -> None:
        self.metadata, self._url = metadata, url
        self._engine: Any = None
        self._sessions: Any = None

    @property
    def engine(self) -> Any:
        """The engine on ``DATABASE_URL``, made on first use with the models' tables created there; raises
        ImportError naming the driver when the URL's isn't installed."""
        if self._engine is None:
            url = self._url()
            if not url:
                raise RuntimeError("a site of SQLAlchemy models keeps its rows in DATABASE_URL, which isn't set")
            sqlalchemy = _sqlalchemy()
            try:
                engine = sqlalchemy.create_engine(url)
            except ImportError as e:
                scheme = sqlalchemy.engine.make_url(url).drivername
                raise ImportError(f"DATABASE_URL's {scheme}:// needs the database driver {e.name or 'it names'}, which "
                                  f"isn't installed; the db extra carries psycopg for postgresql://, and a site installs any "
                                  f"other driver itself (pip install {e.name or 'the driver'})") from e
            self.metadata.create_all(engine)
            self._engine = engine
        return self._engine

    @property
    def sessions(self) -> Any:
        if self._sessions is None:
            from sqlalchemy.orm import sessionmaker

            self._sessions = sessionmaker(bind=self.engine)
        return self._sessions

    def export(self) -> dict:
        sqlalchemy = _sqlalchemy()
        with self.engine.connect() as connection:
            return {table.name: [to_jsonable_python(dict(row._mapping)) for row in connection.execute(
                sqlalchemy.select(table).order_by(*table.primary_key.columns))] for table in self.metadata.sorted_tables}

    def load(self, data: dict) -> None:
        """Insert ``data``'s rows, each table's after the tables it refers to, in their order, each run of rows that name
        the same columns as one insert (one insert takes its columns from its first row, so a row that leaves an
        optional column out goes in its own), leaving out each computed column, which the database works out itself, so
        what ``export`` gave loads back; refuses a name that is no table."""
        tables = {table.name: table for table in self.metadata.sorted_tables}
        unknown = sorted(set(data) - set(tables))
        if unknown:
            raise ValueError(f"no table {', '.join(map(repr, unknown))}; the tables are {sorted(tables)}")
        with self.engine.begin() as connection:
            for name, table in tables.items():
                rows = data.get(name) or []
                if not (isinstance(rows, list) and all(isinstance(row, dict) for row in rows)):
                    raise ValueError(f"table {name!r}'s data is not a list of rows")
                for _, run in itertools.groupby(rows, key=lambda row: sorted(row)):
                    connection.execute(table.insert(), [_row(table, row) for row in run])
            if connection.dialect.name == "postgresql":
                _advance_sequences(connection, tables.values())

    def reset(self) -> None:
        with self.engine.begin() as connection:
            for table in reversed(self.metadata.sorted_tables):
                connection.execute(table.delete())

    def key(self, name: str, row: dict) -> str:
        """A row's identity, its primary key's values, which ``/go``'s diff matches rows by."""
        table = next((t for t in self.metadata.sorted_tables if t.name == name), None)
        columns = [c.name for c in table.primary_key.columns] if table is not None else []
        return json.dumps([row.get(c) for c in columns] if columns else row, sort_keys=True, default=str)


_PARSERS: dict = {datetime.datetime: datetime.datetime.fromisoformat, datetime.date: datetime.date.fromisoformat,
                  datetime.time: datetime.time.fromisoformat, decimal.Decimal: decimal.Decimal, uuid.UUID: uuid.UUID}


def _row(table: Any, row: dict) -> dict:
    """``row`` without its computed columns' values, and with each JSON string a date, time, decimal or UUID column
    holds parsed to that type."""
    parsed = dict(row)
    for column in table.columns:
        if column.computed is not None:
            parsed.pop(column.name, None)
            continue
        value = parsed.get(column.name)
        if isinstance(value, str):
            kind = _python_type(column)
            if kind in _PARSERS:
                parsed[column.name] = _PARSERS[kind](value.replace("Z", "+00:00") if kind is datetime.datetime else value)
    return parsed


def _python_type(column: Any) -> type | None:
    try:
        return column.type.python_type
    except NotImplementedError:
        return None


def _advance_sequences(connection: Any, tables: Iterable[Any]) -> None:
    """Move each serial key's sequence past the rows just loaded, so the next row the site or server adds takes a key
    of its own."""
    sqlalchemy = _sqlalchemy()
    for table in tables:
        keys = list(table.primary_key.columns)
        if len(keys) == 1 and keys[0].autoincrement in (True, "auto") and _python_type(keys[0]) is int:
            quote = connection.dialect.identifier_preparer
            name = quote.format_table(table)
            connection.execute(sqlalchemy.text(
                f"SELECT setval(pg_get_serial_sequence(:table, :column), COALESCE(MAX({quote.quote(keys[0].name)}), 0) + 1, "
                f"false) FROM {name}"), {"table": name, "column": keys[0].name})


class Sessions:
    """A site's ``self.db``: ``self.db()`` is a SQLAlchemy ``Session`` on the site's database, and ``self.db.begin()``
    one in a transaction that commits as its ``with`` ends; the engine is made on first use, not at mount."""

    def __init__(self, tables: Tables) -> None:
        self._tables = tables

    def __call__(self, **options: Any) -> Any:
        return self._tables.sessions(**options)

    def begin(self) -> Any:
        return self._tables.sessions.begin()

    @property
    def engine(self) -> Any:
        return self._tables.engine


class ModelState:
    """A models site's data plane and ``/go``: ``hooks`` loads, exports and resets its data, the state data/add last
    loaded is the initial one, and the data's ``current_user`` is kept beside it."""

    def __init__(self, hooks: Any) -> None:
        self.hooks = hooks
        self.initial: dict = {}
        self.current_user: str | None = None

    async def reset(self) -> None:
        await _maybe_await(self.hooks.reset())
        self.initial, self.current_user = {}, None

    async def load_parts(self, parts: list) -> None:
        for part in parts:
            data = dict(part_data(part))
            if CURRENT_USER in data:
                self.current_user = data.pop(CURRENT_USER)
            await _maybe_await(self.hooks.load(data))
        self.initial = copy.deepcopy(await self.export())

    async def export(self) -> dict:
        data = dict(await _maybe_await(self.hooks.export()))
        if self.current_user is not None:
            data[CURRENT_USER] = self.current_user
        return data

    async def state(self) -> dict:
        key = getattr(self.hooks, "key", None)
        return verifier_state(copy.deepcopy(self.initial), await self.export(), key)
