"""MongoDB implementation of the DocumentStore interface.

Translates the backend-agnostic ``Filter`` / ``UpdateSpec`` / ``Sort`` into
pymongo queries and updates. This is the concrete home for the pymongo idioms
currently spread across the individual ``*/store.py`` modules.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError as PyMongoDuplicateKeyError
from pymongo.errors import OperationFailure

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

logger = logging.getLogger(__name__)


def _field_cond(mongo: dict, field: str) -> dict:
    if field in mongo and not isinstance(mongo[field], dict):
        raise ValueError(f"filter fuses an operator predicate with Eq on field {field!r}; not supported")
    return mongo.setdefault(field, {})


def _apply_predicate(mongo: dict, field: str, pred: Predicate) -> None:
    if isinstance(pred, Eq):
        if isinstance(mongo.get(field), dict):
            raise ValueError(f"filter fuses Eq with an operator predicate on field {field!r}; not supported")
        mongo[field] = pred.value
    elif isinstance(pred, Ne):
        cond = _field_cond(mongo, field)
        cond["$ne"] = pred.value
        cond["$exists"] = True
    elif isinstance(pred, Gte):
        _field_cond(mongo, field)["$gte"] = pred.value
    elif isinstance(pred, Lte):
        _field_cond(mongo, field)["$lte"] = pred.value
    elif isinstance(pred, In):
        _field_cond(mongo, field)["$in"] = list(pred.values)
    elif isinstance(pred, Exists):
        _field_cond(mongo, field)["$exists"] = pred.present
    elif isinstance(pred, LteOrAbsent):
        mongo.setdefault("$and", []).append({"$or": [
            {field: {"$exists": False}},
            {field: None},
            {field: {"$lte": pred.value}},
        ]})
    elif isinstance(pred, AbsentOrNull):
        mongo.setdefault("$and", []).append({"$or": [
            {field: {"$exists": False}},
            {field: None},
        ]})
    else:
        raise TypeError(f"Unsupported predicate: {pred!r}")


def _to_mongo_filter(filter: Filter) -> dict:
    mongo: dict = {}
    for field, predicates in filter.conditions.items():
        for pred in predicates:
            _apply_predicate(mongo, field, pred)
    return mongo


def _to_mongo_sort(sort: Optional[Sort]) -> Optional[list[tuple[str, int]]]:
    if sort is None or not sort.keys:
        return None
    return [(k.field, DESCENDING if k.descending else ASCENDING) for k in sort.keys]


def _to_mongo_update(update: UpdateSpec) -> dict:
    update.validate()
    mongo: dict = {}
    if update.set:
        mongo["$set"] = dict(update.set)
    if update.unset:
        mongo["$unset"] = {path: "" for path in update.unset}
    if update.inc:
        mongo["$inc"] = dict(update.inc)
    if update.add_to_set:
        mongo["$addToSet"] = {p: {"$each": list(v)} for p, v in update.add_to_set.items()}
    if update.push:
        mongo["$push"] = {p: {"$each": list(v)} for p, v in update.push.items()}
    return mongo


@contextmanager
def _unique_violations() -> Iterator[None]:
    """pymongo's duplicate-key error, as the store's own. An upsert can raise it as well as an
    insert: two concurrent upserts both miss, and the second one's insert collides."""
    try:
        yield
    except PyMongoDuplicateKeyError as e:
        raise DuplicateKeyError(str(e)) from e


def _strip_id(doc: Optional[dict]) -> Optional[dict]:
    if doc is not None:
        doc.pop("_id", None)
    return doc


class MongoDocumentStore(DocumentStore):
    """DocumentStore backed by a pymongo ``Database`` (injected, config-free)."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @classmethod
    def from_config(cls, *, uri: str, database: str) -> MongoDocumentStore:
        """Build a pinged, retry-configured MongoClient for ``uri`` and wrap ``database``."""
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=15000,
            socketTimeoutMS=30000,
            retryReads=True,
            retryWrites=True,
        )
        db = client[database]
        try:
            client.admin.command("ping")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e
        logger.info("Connected to MongoDB database: %s", database)
        return cls(db)

    @property
    def database(self) -> Database:
        """The underlying pymongo Database, shared with raw-pymongo callers (Config.db)."""
        return self._db

    def _c(self, collection: str):
        return self._db[collection]

    def mongo_collection(self, collection: str) -> Collection:
        """Raw pymongo Collection escape hatch for consumers (a hosting service, say) that
        need aggregation pipelines the abstract DocumentStore API doesn't expose."""
        return self._c(collection)

    def find_one(
        self, collection: str, filter: Filter, sort: Optional[Sort] = None
    ) -> Optional[dict]:
        doc = self._c(collection).find_one(
            _to_mongo_filter(filter), sort=_to_mongo_sort(sort)
        )
        return _strip_id(doc)

    def find_many_by_id(self, collection: str, id_field: str, ids: list[str]) -> list[dict]:
        found = {}
        for doc in self.query(collection, Filter().where(id_field, In(list(dict.fromkeys(ids))))):
            found.setdefault(doc[id_field], doc)
        return [found[identity] for identity in dict.fromkeys(ids) if identity in found]

    def query(
        self,
        collection: str,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        cursor = self._c(collection).find(_to_mongo_filter(filter))
        mongo_sort = _to_mongo_sort(sort)
        if mongo_sort:
            cursor = cursor.sort(mongo_sort)
        if offset:
            cursor = cursor.skip(offset)
        if limit:
            cursor = cursor.limit(limit)
        return [_strip_id(doc) for doc in cursor]

    def count(self, collection: str, filter: Filter) -> int:
        return self._c(collection).count_documents(_to_mongo_filter(filter))

    def insert(self, collection: str, doc: dict) -> None:
        with _unique_violations():
            self._c(collection).insert_one(dict(doc))

    def update(
        self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool = False
    ) -> int:
        with _unique_violations():
            result = self._c(collection).update_one(
                _to_mongo_filter(filter), _to_mongo_update(update), upsert=upsert
            )
        if result.upserted_id is not None:
            return 1
        return result.matched_count

    def update_one_and_get(
        self,
        collection: str,
        filter: Filter,
        update: UpdateSpec,
        return_after: bool = True,
        upsert: bool = False,
    ) -> Optional[dict]:
        with _unique_violations():
            doc = self._c(collection).find_one_and_update(
                _to_mongo_filter(filter),
                _to_mongo_update(update),
                return_document=ReturnDocument.AFTER if return_after else ReturnDocument.BEFORE,
                upsert=upsert,
            )
        return _strip_id(doc)

    def replace(
        self, collection: str, filter: Filter, doc: dict, upsert: bool = False
    ) -> int:
        with _unique_violations():
            result = self._c(collection).replace_one(
                _to_mongo_filter(filter), dict(doc), upsert=upsert
            )
        if result.upserted_id is not None:
            return 1
        return result.matched_count

    def delete(self, collection: str, filter: Filter) -> int:
        return self._c(collection).delete_one(_to_mongo_filter(filter)).deleted_count

    def ensure_index(
        self,
        collection: str,
        fields: list[str],
        unique: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        kwargs: dict = {}
        if ttl_seconds is not None:
            kwargs["expireAfterSeconds"] = ttl_seconds
        try:
            self._c(collection).create_index(
                [(f, ASCENDING) for f in fields],
                unique=unique,
                name="_".join(fields) + ("_unique" if unique else "_index"),
                **kwargs,
            )
        except OperationFailure as e:
            # 85=IndexOptionsConflict, 86=IndexKeySpecsConflict: an equivalent index
            # already exists under a different name/spec (pre-refactor). Tolerate it.
            if e.code not in (85, 86):
                raise
