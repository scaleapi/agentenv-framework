"""Backend-agnostic document store abstraction."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar, Union

T = TypeVar("T")


# --- Predicates ------------------------------------------------------------


@dataclass(frozen=True)
class Eq:
    """Field equals value."""

    value: Any


@dataclass(frozen=True)
class Ne:
    """Field is present and not equal to value; an absent field does not match."""

    value: Any


@dataclass(frozen=True)
class Gte:
    """Field is greater than or equal to value."""

    value: Any


@dataclass(frozen=True)
class Lte:
    """Field is less than or equal to value."""

    value: Any


@dataclass(frozen=True)
class In:
    """Field is one of values."""

    values: list[Any]


@dataclass(frozen=True)
class Exists:
    """Field is present (present=True) or absent (present=False)."""

    present: bool


@dataclass(frozen=True)
class LteOrAbsent:
    """Field is absent, null, or <= value."""

    value: Any


@dataclass(frozen=True)
class AbsentOrNull:
    """Field is absent or null."""


Predicate = Union[Eq, Ne, Gte, Lte, In, Exists, LteOrAbsent, AbsentOrNull]


@dataclass
class Filter:
    """A flat AND of per-field predicate lists."""

    conditions: dict[str, list[Predicate]] = field(default_factory=dict)

    @classmethod
    def of(cls, **eq: Any) -> "Filter":
        """Build an all-equality filter, e.g. ``Filter.of(id="x", version=3)``."""
        return cls({path: [Eq(value)] for path, value in eq.items()})

    def where(self, field_path: str, predicate: Predicate) -> "Filter":
        """Return a new filter with ``predicate`` added to ``field_path`` (AND)."""
        merged = {path: list(preds) for path, preds in self.conditions.items()}
        merged.setdefault(field_path, []).append(predicate)
        return Filter(merged)


# --- Sort ------------------------------------------------------------------


@dataclass(frozen=True)
class SortKey:
    field: str
    descending: bool = True


@dataclass(frozen=True)
class Sort:
    """An ordered list of sort keys (first key is primary)."""

    keys: tuple[SortKey, ...] = ()

    @classmethod
    def by(cls, field: str, descending: bool = True) -> "Sort":
        return cls((SortKey(field, descending),))


# --- Update ----------------------------------------------------------------


@dataclass
class UpdateSpec:
    """A primitive, backend-agnostic document update; keys are dotted paths."""

    set: dict[str, Any] = field(default_factory=dict)
    unset: set[str] = field(default_factory=lambda: set())
    inc: dict[str, Union[int, float]] = field(default_factory=dict)
    add_to_set: dict[str, list[Any]] = field(default_factory=dict)
    push: dict[str, list[Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.set or self.unset or self.inc or self.add_to_set or self.push)

    def validate(self) -> None:
        """Raise ValueError if a dotted path appears under more than one operator, or if one
        emitted path is an ancestor of another (Mongo rejects both as a path conflict)."""
        seen: dict[str, str] = {}
        conflicts: set[str] = set()
        groups = {
            "set": self.set,
            "unset": self.unset,
            "inc": self.inc,
            "add_to_set": self.add_to_set,
            "push": self.push,
        }
        for op, mapping in groups.items():
            for path in mapping:
                if path in seen:
                    conflicts.add(path)
                seen[path] = op
        if conflicts:
            raise ValueError(
                f"UpdateSpec emits the same path under multiple operators: {sorted(conflicts)}"
            )
        nested = sorted(p for p in seen if any(p.startswith(q + ".") for q in seen))
        if nested:
            raise ValueError(f"UpdateSpec emits a path and one of its ancestors: {nested}")


class DuplicateKeyError(Exception):
    """Raised by ``DocumentStore.insert`` when a unique constraint is violated."""


def _dig(doc: dict, path: str) -> tuple[bool, Any]:
    """Resolve a dotted path in ``doc``; ``(found, value)``, value None when absent."""
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _order_key(value: Any) -> Any:
    """A total-order proxy for a JSON value used as a sort key. ``dict`` and ``list`` are the
    only JSON types Python 3 refuses to ``<``-compare, so collapse them to a deterministic
    string — a same-type sort bucket then never raises ``TypeError``. Scalars pass through so
    their native ordering is preserved (``2 < 10``, not ``"10" < "2"``). ``default=str`` covers
    any non-JSON leaf a backend may hand back (e.g. a BSON datetime)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _reduce_to_latest(docs: list[dict], id_field: str, version_field: str) -> list[dict]:
    """Keep the highest-``version_field`` doc per ``id_field``; skip docs with no id."""
    best: dict[Any, tuple[tuple[bool, Any], dict]] = {}
    for doc in docs:
        found, key = _dig(doc, id_field)
        if not found:
            continue
        has_version, version = _dig(doc, version_field)
        rank = (has_version and version is not None, version)  # missing version sorts lowest
        current = best.get(key)
        if current is None or rank > current[0]:
            best[key] = (rank, doc)
    return [doc for _, doc in best.values()]


def _apply_sort(docs: list[dict], sort: Optional[Sort]) -> list[dict]:
    """Apply ``sort`` to an already-materialized list; absent values order last,
    independent of each key's direction. The sort key is ``(type_name, order_key)``: the
    type name keeps mixed-type values across documents (an int in one, a str in another)
    from ever reaching a ``<`` (Python 3 refuses to compare unlike types), and ``_order_key``
    keeps same-type-but-uncomparable values (two dicts, two lists) from raising too —
    within a type, by value."""
    if not sort or not sort.keys:
        return docs
    ordered = list(docs)
    for key in reversed(sort.keys):
        present, absent = [], []
        for doc in ordered:
            found, value = _dig(doc, key.field)
            (present if found and value is not None else absent).append(doc)
        present.sort(
            key=lambda d, f=key.field: (type(v := _dig(d, f)[1]).__name__, _order_key(v)),
            reverse=key.descending,
        )
        ordered = present + absent
    return ordered


class DocumentStore(ABC):
    """A backend-agnostic document store over named collections; backend id fields never leak."""

    @classmethod
    def from_config(cls, **config) -> DocumentStore:
        """Construct from a resolved config table; backends that build a client override this."""
        return cls(**config)

    @abstractmethod
    def find_one(
        self, collection: str, filter: Filter, sort: Optional[Sort] = None
    ) -> Optional[dict]:
        """Return the first matching document (per ``sort``), or None."""

    @abstractmethod
    def query(
        self,
        collection: str,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        """Return all matching documents, applying sort then offset then limit."""

    @abstractmethod
    def count(self, collection: str, filter: Filter) -> int:
        """Return the number of matching documents."""

    def latest_per_id(
        self,
        collection: str,
        filter: Filter,
        *,
        id_field: str = "id",
        version_field: str = "version",
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        """The highest-``version_field`` document for each distinct ``id_field`` — the
        "latest version of each entity" read a list endpoint needs.

        ``filter`` applies before grouping (it selects among all versions, then reduces
        to the newest per id); ``sort`` / ``offset`` / ``limit`` apply after, to the
        reduced set. Documents missing ``id_field`` are skipped; a missing
        ``version_field`` sorts lowest. Reduces in Python over `query()`, so any
        backend gets it without extra work.
        """
        docs = _reduce_to_latest(self.query(collection, filter), id_field, version_field)
        docs = _apply_sort(docs, sort)
        if offset:
            docs = docs[offset:]
        if limit:
            docs = docs[:limit]
        return docs

    def count_distinct(self, collection: str, filter: Filter, *, id_field: str = "id") -> int:
        """Number of distinct ``id_field`` values among matching documents.

        The pagination total that pairs with `latest_per_id()` — a row count would
        over-report, since one entity contributes one row per version.
        """
        keys = set()
        for doc in self.query(collection, filter):
            found, value = _dig(doc, id_field)
            if found:
                keys.add(value)
        return len(keys)

    @abstractmethod
    def insert(self, collection: str, doc: dict) -> None:
        """Insert one document. Raises DuplicateKeyError on unique violation."""

    @abstractmethod
    def update(
        self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool = False
    ) -> int:
        """Apply ``update`` to the first match; return matched count (0/1), where 0 is the CAS miss. ``upsert=True`` synthesizes a doc from the filter on no match."""

    @abstractmethod
    def update_one_and_get(
        self,
        collection: str,
        filter: Filter,
        update: UpdateSpec,
        return_after: bool = True,
        upsert: bool = False,
    ) -> Optional[dict]:
        """Apply ``update`` to the first match; return the doc (post-update if ``return_after`` else pre-update), or None on no match."""

    @abstractmethod
    def replace(
        self, collection: str, filter: Filter, doc: dict, upsert: bool = False
    ) -> int:
        """Replace the first matching document wholesale; return matched count (0/1)."""

    @abstractmethod
    def delete(self, collection: str, filter: Filter) -> int:
        """Delete the first matching document; return deleted count (0/1)."""

    @abstractmethod
    def ensure_index(
        self,
        collection: str,
        fields: list[str],
        unique: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Register an index over ``fields`` (idempotent); ``unique=True`` is enforced by ``insert`` via ``DuplicateKeyError``. ``ttl_seconds`` requests server-side expiry off the single indexed datetime field (backends without TTL may ignore it)."""


RESERVED_ID_PREFIX = "@"


def reject_reserved_id(entity_id: object) -> None:
    """Refuse an id claiming the ``@`` prefix, which opens a registry-qualified
    ``@namespace/id``.

    Reserved while agent-env is still private: once third parties have created ids, taking
    the prefix back stops being free. No stored document starts with it.
    """
    if isinstance(entity_id, str) and entity_id.startswith(RESERVED_ID_PREFIX):
        raise ValueError(
            f"id {entity_id!r} starts with the reserved {RESERVED_ID_PREFIX!r} prefix: that "
            "addresses a registry-qualified name (@namespace/id), so a local id cannot use it"
        )


class VersionedEntityStore(Generic[T]):
    """The shared ``(id, version)`` versioning logic on top of a DocumentStore."""

    def __init__(
        self,
        docs: DocumentStore,
        collection: str,
        serialize: Optional[Callable[[T], dict]],
        deserialize: Callable[[dict], T],
        secondary_indexes: Optional[list[list[str]]] = None,
    ) -> None:
        self._doc_store = docs
        self._collection = collection
        self._serialize = serialize
        self._deserialize = deserialize
        self._doc_store.ensure_index(collection, ["id", "version"], unique=True)
        for fields in secondary_indexes or []:
            self._doc_store.ensure_index(collection, fields)

    def get(self, id: str, version: Optional[int] = None) -> Optional[T]:
        if version is not None:
            doc = self._doc_store.find_one(self._collection, Filter.of(id=id, version=version))
        else:
            doc = self._doc_store.find_one(
                self._collection, Filter.of(id=id), sort=Sort.by("version", descending=True)
            )
        return self._deserialize(doc) if doc is not None else None

    def next_version(self, id: str) -> int:
        """Allocate the version a write to ``id`` would land on.

        The reservation is enforced here and not only at ``put`` because every artifact
        helper calls this first and then writes remote data — an object to S3, an image to a
        registry — before it has a document to store. Refusing the id at the end would leave
        that data orphaned with no artifact record pointing at it.
        """
        reject_reserved_id(id)
        doc = self._doc_store.find_one(
            self._collection, Filter.of(id=id), sort=Sort.by("version", descending=True)
        )
        return (doc["version"] + 1) if doc is not None else 1

    def put(self, entity: T, max_retries: int = 5) -> int:
        """Serialize and insert the entity under the next free version; return that version."""
        if self._serialize is None:
            raise RuntimeError("VersionedEntityStore was built without a serializer; put() is unsupported")
        doc = self._serialize(entity)
        reject_reserved_id(doc.get("id"))
        last_error: Optional[DuplicateKeyError] = None
        for _ in range(max_retries):
            version = self.next_version(doc["id"])
            attempt = {**doc, "version": version}
            try:
                self._doc_store.insert(self._collection, attempt)
                return version
            except DuplicateKeyError as e:
                last_error = e
        raise DuplicateKeyError(
            f"Could not assign a version for id={doc.get('id')!r} after {max_retries} retries"
        ) from last_error

    def query(
        self,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[T]:
        return [
            self._deserialize(doc)
            for doc in self._doc_store.query(self._collection, filter, sort, limit, offset)
        ]

    def count(self, filter: Filter) -> int:
        return self._doc_store.count(self._collection, filter)


class VersionedEntityStoreCache(Generic[T]):
    """The ``VersionedEntityStore`` for whichever backend is current.

    Building one runs ``ensure_index``, so it is cached rather than rebuilt per call; but one
    built on a previous backend goes on writing there after ``reset_config()`` re-points, so
    the cache is keyed on the backend instead of filled once.

    Lock-free, and deliberately so: constructing the view does I/O, which is not something to
    hold a lock across. The backend and the view built for it are one tuple rebound in a
    single assignment, so they cannot be seen apart; and ``for_store`` returns the view it
    built rather than re-reading the field, so a builder that loses the race still hands its
    caller a view on the backend that caller asked for. The loser's only cost is a redundant
    ``ensure_index``, which is idempotent.
    """

    def __init__(
        self,
        collection: str,
        serialize: Optional[Callable[[T], dict]],
        deserialize: Callable[[dict], T],
        secondary_indexes: Optional[list[list[str]]] = None,
    ) -> None:
        self._collection = collection
        self._serialize = serialize
        self._deserialize = deserialize
        self._secondary_indexes = secondary_indexes
        self._cached: Optional[tuple[DocumentStore, VersionedEntityStore[T]]] = None

    def for_store(self, store: DocumentStore) -> VersionedEntityStore[T]:
        cached = self._cached
        if cached is not None and cached[0] is store:
            return cached[1]
        view: VersionedEntityStore[T] = VersionedEntityStore(
            store, self._collection, self._serialize, self._deserialize,
            secondary_indexes=self._secondary_indexes,
        )
        self._cached = (store, view)
        return view


def rev_precondition(doc: dict, counter_field: str) -> Predicate:
    """CAS predicate for ``doc``'s current ``counter_field``; ``Exists(False)`` when absent so the first write migrates a pre-counter doc rather than dead-looping on ``Eq(0)``."""
    current = doc.get(counter_field)
    return Eq(current) if current is not None else Exists(present=False)


def compare_and_swap(
    docs: DocumentStore,
    collection: str,
    key: Filter,
    mutate: Callable[[dict], Optional[UpdateSpec]],
    *,
    counter_field: str,
    max_attempts: int = 1000,
) -> Optional[dict]:
    """Read the doc matched by ``key``, apply ``mutate`` under an optimistic CAS on ``counter_field``, retrying on contention; return the post-update doc (or None if ``key`` misses or ``mutate`` returns None). ``mutate`` must not write ``counter_field`` — the increment is managed here."""
    for _ in range(max_attempts):
        doc = docs.find_one(collection, key)
        if doc is None:
            return None
        spec = mutate(doc)
        if spec is None:
            return None
        spec.inc = {**spec.inc, counter_field: 1}
        result = docs.update_one_and_get(
            collection, key.where(counter_field, rev_precondition(doc, counter_field)), spec
        )
        if result is not None:
            return result
    raise RuntimeError(
        f"compare_and_swap: did not converge for collection={collection} key={key.conditions}"
    )
