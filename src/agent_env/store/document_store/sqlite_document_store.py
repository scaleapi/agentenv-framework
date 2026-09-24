"""SQLite implementation of the DocumentStore interface (stdlib, no infra).

Stores each document as a JSON row in a per-collection table; SQLite provides
uniqueness (via unique indexes on ``json_extract``), atomicity (transactions),
and durability. ``Filter`` / ``UpdateSpec`` / ``Sort`` semantics are evaluated in
Python, mirroring ``mongo_document_store`` as the oracle. Datetimes are
serialized as ISO strings.

SQL safety note: every user-controlled *value* is bound as a ``?`` placeholder.
The only interpolated tokens are SQL *identifiers* (table / index / column
names), which cannot be passed as bound parameters. Those are derived from a
strict allowlist (see ``_safe_ident`` / ``_safe_path``) before interpolation, so
the ``execute(f"...")`` sites carry ``# nosemgrep`` for the raw-query rule.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from agent_env.store.document_store.document_store import (
    AbsentOrNull,
    DocumentStore,
    DuplicateKeyError,
    Eq,
    Exists,
    Filter,
    Gte,
    In,
    Lte,
    LteOrAbsent,
    Ne,
    Predicate,
    Sort,
    UpdateSpec,
)
from agent_env.store.local_state import ensure_state_dir

# SQL identifiers (table/index names) must be a bare word; JSON-path field
# references additionally allow dots for nested paths (e.g. ``metadata.rev``).
_IDENT_RE = re.compile(r"[A-Za-z0-9_]+")
_PATH_RE = re.compile(r"[A-Za-z0-9_.]+")


class LocalSqliteDocumentStore(DocumentStore):
    """DocumentStore backed by a single stdlib-``sqlite3`` database file."""

    def __init__(self, path: str) -> None:
        # Nothing touches disk until the first operation: constructing a store creates nothing.
        self._path = Path(path)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._tables: set[str] = set()

    @property
    def _conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is None:
                ensure_state_dir(self._path.parent)
                conn = sqlite3.connect(str(self._path), check_same_thread=False, isolation_level=None)
                conn.execute("PRAGMA journal_mode=WAL")
                self._connection = conn
            return self._connection

    def find_one(
        self, collection: str, filter: Filter, sort: Optional[Sort] = None
    ) -> Optional[dict]:
        with self._lock:
            matches = [doc for _, doc in self._load(collection) if self._matches(doc, filter)]
            matches = self._sorted(matches, sort)
            return matches[0] if matches else None

    def query(
        self,
        collection: str,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        with self._lock:
            docs = [doc for _, doc in self._load(collection) if self._matches(doc, filter)]
            docs = self._sorted(docs, sort)
            if offset:
                docs = docs[offset:]
            if limit:
                docs = docs[:limit]
            return docs

    def count(self, collection: str, filter: Filter) -> int:
        with self._lock:
            return sum(1 for _, doc in self._load(collection) if self._matches(doc, filter))

    def insert(self, collection: str, doc: dict) -> None:
        with self._lock:
            tbl = self._ensure_table(collection)
            self._insert_raw(tbl, dict(doc))

    def update(
        self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool = False
    ) -> int:
        update.validate()
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is not None:
                rowid, doc = match
                self._apply_update(doc, update)
                self._conn.execute(
                    # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; doc/rowid are bound
                    f'UPDATE "{self._table(collection)}" SET doc=? WHERE rowid=?',
                    (self._dumps(doc), rowid),
                )
                return 1
            if upsert:
                self._insert_raw(self._ensure_table(collection), self._synthesize(filter, update))
                return 1
            return 0

    def update_one_and_get(
        self,
        collection: str,
        filter: Filter,
        update: UpdateSpec,
        return_after: bool = True,
        upsert: bool = False,
    ) -> Optional[dict]:
        update.validate()
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is not None:
                rowid, doc = match
                before = json.loads(self._dumps(doc))
                self._apply_update(doc, update)
                self._conn.execute(
                    # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; doc/rowid are bound
                    f'UPDATE "{self._table(collection)}" SET doc=? WHERE rowid=?',
                    (self._dumps(doc), rowid),
                )
                return doc if return_after else before
            if upsert:
                new_doc = self._synthesize(filter, update)
                self._insert_raw(self._ensure_table(collection), new_doc)
                return new_doc if return_after else None
            return None

    def replace(
        self, collection: str, filter: Filter, doc: dict, upsert: bool = False
    ) -> int:
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is not None:
                rowid, _ = match
                self._conn.execute(
                    # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; doc/rowid are bound
                    f'UPDATE "{self._table(collection)}" SET doc=? WHERE rowid=?',
                    (self._dumps(dict(doc)), rowid),
                )
                return 1
            if upsert:
                self._insert_raw(self._ensure_table(collection), dict(doc))
                return 1
            return 0

    def delete(self, collection: str, filter: Filter) -> int:
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is None:
                return 0
            rowid, _ = match
            # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; rowid is bound
            self._conn.execute(f'DELETE FROM "{self._table(collection)}" WHERE rowid=?', (rowid,))
            return 1

    def ensure_index(
        self,
        collection: str,
        fields: list[str],
        unique: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if ttl_seconds is not None:
            logger.warning(
                "LocalSqliteDocumentStore has no TTL support; ttl_seconds=%s on %s%s ignored — rows will not auto-expire.",
                ttl_seconds, collection, fields,
            )
        with self._lock:
            tbl = self._ensure_table(collection)
            raw_name = tbl + "_" + "_".join(fields) + ("_unique" if unique else "_index")
            name = self._safe_ident(re.sub(r"\W", "_", raw_name))
            cols = ", ".join(f"json_extract(doc, '$.{self._safe_path(f)}')" for f in fields)
            uniq = "UNIQUE " if unique else ""
            self._conn.execute(
                # nosemgrep: sqlalchemy-execute-raw-query -- name/tbl/cols are validated identifiers
                f'CREATE {uniq}INDEX IF NOT EXISTS "{name}" ON "{tbl}" ({cols})'
            )

    # --- sqlite plumbing ---

    @staticmethod
    def _safe_ident(name: str) -> str:
        """Validate a SQL identifier (table/index name) against a strict allowlist.

        Identifiers cannot be passed as bound parameters, so we assert they are a
        bare ``[A-Za-z0-9_]`` word before interpolating them into SQL. User input
        only ever reaches SQL as a bound ``?`` value, never as an identifier.
        """
        if not _IDENT_RE.fullmatch(name):
            raise ValueError(f"unsafe SQL identifier: {name!r}")
        return name

    @staticmethod
    def _safe_path(field: str) -> str:
        """Validate a JSON-path field reference used inside ``json_extract('$.…')``."""
        if not _PATH_RE.fullmatch(field):
            raise ValueError(f"unsafe JSON field path: {field!r}")
        return field

    def _table(self, collection: str) -> str:
        # The table name is an identifier (not bindable): sanitize the collection
        # to a bare word, then assert the allowlist as defense-in-depth.
        return self._safe_ident("docs_" + re.sub(r"\W", "_", collection))

    def _ensure_table(self, collection: str) -> str:
        tbl = self._table(collection)
        if tbl not in self._tables:
            # nosemgrep: sqlalchemy-execute-raw-query -- tbl is a validated identifier
            self._conn.execute(f'CREATE TABLE IF NOT EXISTS "{tbl}" (doc TEXT NOT NULL)')
            self._tables.add(tbl)
        return tbl

    def _adopt_if_created(self, tbl: str) -> bool:
        """Adopt a table that exists on disk but is not yet in ``self._tables``."""
        if self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
        ).fetchone():
            self._tables.add(tbl)
            return True
        return False

    def _load(self, collection: str) -> list[tuple[int, dict]]:
        tbl = self._table(collection)
        if tbl not in self._tables and not self._adopt_if_created(tbl):
            return []
        return [
            (row[0], json.loads(row[1]))
            # nosemgrep: sqlalchemy-execute-raw-query -- tbl is a validated identifier
            for row in self._conn.execute(f'SELECT rowid, doc FROM "{tbl}"')
        ]

    def _first_match(self, collection: str, filter: Filter) -> Optional[tuple[int, dict]]:
        for rowid, doc in self._load(collection):
            if self._matches(doc, filter):
                return rowid, doc
        return None

    @contextmanager
    def _immediate(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def _insert_raw(self, tbl: str, doc: dict) -> None:
        try:
            # nosemgrep: sqlalchemy-execute-raw-query -- tbl is a validated identifier; doc is bound
            self._conn.execute(f'INSERT INTO "{tbl}"(doc) VALUES(?)', (self._dumps(doc),))
        except sqlite3.IntegrityError as e:
            raise DuplicateKeyError(str(e)) from e

    def _json_default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")

    def _dumps(self, doc: dict) -> str:
        return json.dumps(doc, default=self._json_default)

    # --- filter / sort evaluation (Mongo array semantics) ---

    def _resolve(self, doc: dict, path: str) -> tuple[bool, list]:
        """Resolve a dotted path to (present, values): a numeric segment indexes an
        array; a non-numeric segment into an array traverses its elements."""
        cursors = [doc]
        for seg in path.split("."):
            nxt = []
            for cur in cursors:
                if isinstance(cur, dict):
                    if seg in cur:
                        nxt.append(cur[seg])
                elif isinstance(cur, list):
                    if seg.isdigit():
                        idx = int(seg)
                        if 0 <= idx < len(cur):
                            nxt.append(cur[idx])
                    else:
                        for el in cur:
                            if isinstance(el, dict) and seg in el:
                                nxt.append(el[seg])
            cursors = nxt
            if not cursors:
                return False, []
        return True, cursors

    def _eq(self, rv, value) -> bool:
        return rv == value or (isinstance(rv, list) and value in rv)

    def _cmp(self, rv, value, op: str) -> bool:
        try:
            return rv >= value if op == "ge" else rv <= value
        except TypeError:
            return False

    def _pred_matches(self, doc: dict, field: str, pred: Predicate) -> bool:
        present, values = self._resolve(doc, field)
        if isinstance(pred, Eq):
            return any(self._eq(rv, pred.value) for rv in values) if present else pred.value is None
        if isinstance(pred, Ne):
            return present and not any(self._eq(rv, pred.value) for rv in values)
        if isinstance(pred, In):
            return present and any(
                rv in pred.values or (isinstance(rv, list) and any(e in pred.values for e in rv))
                for rv in values
            )
        if isinstance(pred, Gte):
            return any(self._cmp(rv, pred.value, "ge") for rv in values)
        if isinstance(pred, Lte):
            return any(self._cmp(rv, pred.value, "le") for rv in values)
        if isinstance(pred, Exists):
            return present == pred.present
        if isinstance(pred, LteOrAbsent):
            if not present:
                return True
            return any(rv is None for rv in values) or any(
                self._cmp(rv, pred.value, "le") for rv in values
            )
        if isinstance(pred, AbsentOrNull):
            return (not present) or any(rv is None for rv in values)
        raise TypeError(f"Unsupported predicate: {pred!r}")

    def _matches(self, doc: dict, filter: Filter) -> bool:
        return all(
            self._pred_matches(doc, field, pred)
            for field, preds in filter.conditions.items()
            for pred in preds
        )

    def _sort_key(self, doc: dict, field: str):
        present, values = self._resolve(doc, field)
        if not present or values[0] is None:
            return (0,)
        return (1, values[0])

    def _sorted(self, docs: list[dict], sort: Optional[Sort]) -> list[dict]:
        if sort is None or not sort.keys:
            return docs
        for key in reversed(sort.keys):
            docs = sorted(docs, key=lambda d, k=key: self._sort_key(d, k.field), reverse=key.descending)
        return docs

    # --- update application ---

    def _get_child(self, container, seg: str):
        if isinstance(container, dict):
            return container.get(seg)
        if isinstance(container, list) and seg.isdigit() and 0 <= int(seg) < len(container):
            return container[int(seg)]
        return None

    def _step_write(self, container, seg: str):
        if isinstance(container, list):
            return container[int(seg)]
        child = container.get(seg)
        if not isinstance(child, (dict, list)):
            child = {}
            container[seg] = child
        return child

    def _assign(self, container, seg: str, value) -> None:
        if isinstance(container, list):
            container[int(seg)] = value
        else:
            container[seg] = value

    def _set_path(self, doc: dict, path: str, value) -> None:
        segs = path.split(".")
        cur = doc
        for seg in segs[:-1]:
            cur = self._step_write(cur, seg)
        self._assign(cur, segs[-1], value)

    def _unset_path(self, doc: dict, path: str) -> None:
        segs = path.split(".")
        cur = doc
        for seg in segs[:-1]:
            cur = self._get_child(cur, seg)
            if cur is None:
                return
        last = segs[-1]
        if isinstance(cur, dict):
            cur.pop(last, None)
        elif isinstance(cur, list) and last.isdigit() and 0 <= int(last) < len(cur):
            cur[int(last)] = None

    def _ensure_list(self, doc: dict, path: str) -> list:
        segs = path.split(".")
        cur = doc
        for seg in segs[:-1]:
            cur = self._step_write(cur, seg)
        existing = self._get_child(cur, segs[-1])
        if not isinstance(existing, list):
            existing = []
            self._assign(cur, segs[-1], existing)
        return existing

    def _apply_update(self, doc: dict, spec: UpdateSpec) -> None:
        spec.validate()
        for path, value in spec.set.items():
            self._set_path(doc, path, value)
        for path in spec.unset:
            self._unset_path(doc, path)
        for path, delta in spec.inc.items():
            segs = path.split(".")
            cur = doc
            for seg in segs[:-1]:
                cur = self._step_write(cur, seg)
            current = self._get_child(cur, segs[-1])
            self._assign(cur, segs[-1], (current or 0) + delta)
        for path, items in spec.add_to_set.items():
            lst = self._ensure_list(doc, path)
            for it in items:
                if it not in lst:
                    lst.append(it)
        for path, items in spec.push.items():
            self._ensure_list(doc, path).extend(items)

    def _synthesize(self, filter: Filter, spec: UpdateSpec) -> dict:
        doc: dict = {}
        for field, preds in filter.conditions.items():
            for pred in preds:
                if isinstance(pred, Eq):
                    self._set_path(doc, field, pred.value)
        self._apply_update(doc, spec)
        return doc
