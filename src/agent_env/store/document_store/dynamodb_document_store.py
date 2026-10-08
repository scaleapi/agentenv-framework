"""DynamoDB DocumentStore: one on-demand table per collection, one JSON blob per item. The sort key's
name records the collection's unique index (``sk`` when it has none), so the table's own key schema fixes
how every process keys items. Filters run in Python (``evaluation``); writes are conditional on the blob read."""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

import boto3

from agent_env.store.base import ConcurrentModificationError
from agent_env.store.document_store import evaluation
from agent_env.store.document_store.document_store import (
    DocumentStore,
    DuplicateKeyError,
    Eq,
    Filter,
    Sort,
    UpdateSpec,
)

logger = logging.getLogger(__name__)

_CAS_ATTEMPTS = 64
_CAS_BACKOFF_SECONDS = 0.005
_CAS_MAX_BACKOFF_SECONDS = 0.25
_TABLE_POLL_SECONDS = 1
_TABLE_POLL_ATTEMPTS = 120
_UNINDEXED_SORT_KEY = "sk"
_BATCH_GET_MAX_KEYS = 100


def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")


def _dumps(value) -> str:
    return json.dumps(value, default=_json_default)


def _unchanged(blob: str) -> dict:
    return {
        "ConditionExpression": "#d = :old",
        "ExpressionAttributeNames": {"#d": "doc"},
        "ExpressionAttributeValues": {":old": {"S": blob}},
    }


class DynamoDbDocumentStore(DocumentStore):
    """DocumentStore backed by DynamoDB tables named ``{table_prefix}{collection}``."""

    def __init__(self, table_prefix: str = "", region: Optional[str] = None) -> None:
        self._table_prefix = table_prefix
        self._client = boto3.client("dynamodb", region_name=region)
        self._sort_keys: dict[str, str] = {}

    def _table_name(self, collection: str) -> str:
        return f"{self._table_prefix}{collection}"

    def _key_fields(self, collection: str) -> Optional[list[str]]:
        sort_key = self._sort_keys[collection]
        return None if sort_key == _UNINDEXED_SORT_KEY else json.loads(sort_key)

    def _wait(self, name: str) -> None:
        self._client.get_waiter("table_exists").wait(
            TableName=name, WaiterConfig={"Delay": _TABLE_POLL_SECONDS, "MaxAttempts": _TABLE_POLL_ATTEMPTS}
        )

    def _table_exists(self, collection: str) -> bool:
        """Whether the table exists; the first time, waits out its creation and learns its sort key."""
        if collection in self._sort_keys:
            return True
        name = self._table_name(collection)
        try:
            table = self._client.describe_table(TableName=name)["Table"]
        except self._client.exceptions.ResourceNotFoundException:
            return False
        if table["TableStatus"] == "CREATING":
            self._wait(name)
        self._sort_keys[collection] = next(k["AttributeName"] for k in table["KeySchema"] if k["KeyType"] == "RANGE")
        return True

    def _table(self, collection: str, key_fields: Optional[list[str]] = None) -> str:
        """The collection's table name, creating the table on first use with a sort key named for ``key_fields``."""
        name = self._table_name(collection)
        if not self._table_exists(collection):
            sort_key = json.dumps(key_fields) if key_fields else _UNINDEXED_SORT_KEY
            try:
                self._client.create_table(
                    TableName=name,
                    KeySchema=[
                        {"AttributeName": "pk", "KeyType": "HASH"},
                        {"AttributeName": sort_key, "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "pk", "AttributeType": "S"},
                        {"AttributeName": sort_key, "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
            except self._client.exceptions.ResourceInUseException:
                pass
            self._wait(name)
            self._table_exists(collection)
        return name

    def _key(self, collection: str, doc: dict, current: Optional[dict] = None) -> dict:
        """``doc``'s item key: its unique-index values, else ``current`` or a generated one."""
        sort_key = self._sort_keys[collection]
        fields = self._key_fields(collection)
        if fields is None:
            return current or {"pk": {"S": uuid.uuid4().hex}, sort_key: {"S": "[]"}}
        values = [doc.get(f) for f in fields]
        return {"pk": {"S": _dumps(values[0])}, sort_key: {"S": _dumps(values[1:])}}

    def _candidates(self, collection: str, filter: Filter) -> list[tuple[dict, str, dict]]:
        """``(key, stored blob, decoded doc)`` for every item that might match ``filter``: a GetItem when it
        pins every key field, a Query when it pins the first, else a Scan."""
        if not self._table_exists(collection):
            return []
        sort_key = self._sort_keys[collection]
        fields = self._key_fields(collection) or []
        pinned = {
            f: eq[0] for f in fields if (eq := [p.value for p in filter.conditions.get(f, []) if isinstance(p, Eq)])
        }
        kwargs: dict = {"TableName": self._table_name(collection), "ConsistentRead": True}
        if fields and len(pinned) == len(fields):
            item = self._client.get_item(Key=self._key(collection, pinned), **kwargs).get("Item")
            items = [item] if item else []
        else:
            if fields and fields[0] in pinned:
                kwargs["KeyConditionExpression"] = "pk = :pk"
                kwargs["ExpressionAttributeValues"] = {":pk": {"S": _dumps(pinned[fields[0]])}}
            op = "query" if "KeyConditionExpression" in kwargs else "scan"
            items = (it for page in self._client.get_paginator(op).paginate(**kwargs) for it in page["Items"])
        return [({"pk": it["pk"], sort_key: it[sort_key]}, it["doc"]["S"], json.loads(it["doc"]["S"])) for it in items]

    def _matching(self, collection: str, filter: Filter) -> list[dict]:
        return [doc for _, _, doc in self._candidates(collection, filter) if evaluation.matches(doc, filter)]

    def _put_new(self, collection: str, doc: dict) -> None:
        name = self._table(collection)
        key = self._key(collection, doc)
        try:
            self._client.put_item(
                TableName=name,
                Item={**key, "doc": {"S": _dumps(doc)}},
                ConditionExpression="attribute_not_exists(pk)",
            )
        except self._client.exceptions.ConditionalCheckFailedException as e:
            raise DuplicateKeyError(f"duplicate key in {collection!r}: {[v['S'] for v in key.values()]}") from e

    def _store(self, collection: str, key: dict, blob: str, doc: dict) -> None:
        """Write ``doc`` over the item read as ``(key, blob)``."""
        if self._key(collection, doc, current=key) != key:
            raise NotImplementedError(
                f"DynamoDbDocumentStore cannot change a unique-index field of a stored {collection!r} document"
            )
        self._client.put_item(
            TableName=self._table_name(collection), Item={**key, "doc": {"S": _dumps(doc)}}, **_unchanged(blob)
        )

    def _cas(self, collection: str, filter: Filter, write: Callable[[dict, str, dict], Any]) -> Any:
        """``write(key, blob, doc)`` on the first match, re-read after a jittered backoff when a concurrent
        writer changed it first; None when nothing matches. A conflict means another write landed."""
        for attempt in range(_CAS_ATTEMPTS):
            match = next(
                (c for c in self._candidates(collection, filter) if evaluation.matches(c[2], filter)), None
            )
            if match is None:
                return None
            try:
                return write(*match)
            except self._client.exceptions.ConditionalCheckFailedException:
                time.sleep(random.uniform(0, min(_CAS_MAX_BACKOFF_SECONDS, _CAS_BACKOFF_SECONDS * 2**attempt)))
        raise ConcurrentModificationError(
            f"{collection!r}: the matched document changed on each of {_CAS_ATTEMPTS} attempts"
        )

    def _insert_or_write(self, collection: str, filter: Filter, doc: dict, write: Callable[[dict, str, dict], Any]) -> Any:
        """Insert ``doc``, or, when a concurrent upsert inserted it first, ``write`` the winner; None when
        inserted. A key still holding a document ``filter`` doesn't match stays a ``DuplicateKeyError``;
        a winner deleted before the re-read frees the key, so the insert runs again."""
        for _ in range(_CAS_ATTEMPTS):
            try:
                self._put_new(collection, doc)
                return None
            except DuplicateKeyError:
                result = self._cas(collection, filter, write)
                if result is not None:
                    return result
                held = self._client.get_item(
                    TableName=self._table_name(collection), Key=self._key(collection, doc), ConsistentRead=True
                )
                if "Item" in held:
                    raise
        raise ConcurrentModificationError(
            f"{collection!r}: the upserted key was taken and freed on each of {_CAS_ATTEMPTS} attempts"
        )

    def find_one(self, collection: str, filter: Filter, sort: Optional[Sort] = None) -> Optional[dict]:
        docs = evaluation.sort_docs(self._matching(collection, filter), sort)
        return docs[0] if docs else None

    def find_many_by_id(self, collection: str, id_field: str, ids: list[str]) -> list[dict]:
        identities = list(dict.fromkeys(ids))
        if not identities or not self._table_exists(collection):
            return []
        if self._key_fields(collection) != [id_field]:
            return super().find_many_by_id(collection, id_field, identities)
        name = self._table_name(collection)
        found = {}
        for start in range(0, len(identities), _BATCH_GET_MAX_KEYS):
            keys = [self._key(collection, {id_field: identity}) for identity in identities[start:start + _BATCH_GET_MAX_KEYS]]
            pending = {name: {"Keys": keys, "ConsistentRead": True}}
            for attempt in range(_CAS_ATTEMPTS):
                result = self._client.batch_get_item(RequestItems=pending)
                for item in result.get("Responses", {}).get(name, []):
                    doc = json.loads(item["doc"]["S"])
                    found.setdefault(doc[id_field], doc)
                pending = result.get("UnprocessedKeys", {})
                if not pending:
                    break
                time.sleep(random.uniform(0, min(_CAS_MAX_BACKOFF_SECONDS, _CAS_BACKOFF_SECONDS * 2**attempt)))
            else:
                raise TimeoutError(f"{collection!r}: DynamoDB batch read left unprocessed keys after {_CAS_ATTEMPTS} attempts")
        return [found[identity] for identity in identities if identity in found]

    def query(
        self,
        collection: str,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        docs = evaluation.sort_docs(self._matching(collection, filter), sort)
        if offset:
            docs = docs[offset:]
        if limit:
            docs = docs[:limit]
        return docs

    def count(self, collection: str, filter: Filter) -> int:
        return len(self._matching(collection, filter))

    def insert(self, collection: str, doc: dict) -> None:
        self._put_new(collection, dict(doc))

    def update(self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool = False) -> int:
        return int(self.update_one_and_get(collection, filter, update, return_after=True, upsert=upsert) is not None)

    def update_one_and_get(
        self,
        collection: str,
        filter: Filter,
        update: UpdateSpec,
        return_after: bool = True,
        upsert: bool = False,
    ) -> Optional[dict]:
        update.validate()

        def write(key: dict, blob: str, doc: dict) -> dict:
            evaluation.apply_update(doc, update)
            self._store(collection, key, blob, doc)
            return doc if return_after else json.loads(blob)

        result = self._cas(collection, filter, write)
        if result is not None:
            return result
        if upsert:
            new_doc = evaluation.synthesize(filter, update)
            raced = self._insert_or_write(collection, filter, new_doc, write)
            if raced is not None:
                return raced
            return new_doc if return_after else None
        return None

    def replace(self, collection: str, filter: Filter, doc: dict, upsert: bool = False) -> int:
        new_doc = dict(doc)

        def write(key: dict, blob: str, _old: dict) -> int:
            self._store(collection, key, blob, new_doc)
            return 1

        if self._cas(collection, filter, write):
            return 1
        if upsert:
            self._insert_or_write(collection, filter, new_doc, write)
            return 1
        return 0

    def delete(self, collection: str, filter: Filter) -> int:
        def write(key: dict, blob: str, _doc: dict) -> int:
            self._client.delete_item(TableName=self._table_name(collection), Key=key, **_unchanged(blob))
            return 1

        return self._cas(collection, filter, write) or 0

    def ensure_index(
        self,
        collection: str,
        fields: list[str],
        unique: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """A unique index becomes the collection's item key; other indexes are not built."""
        if ttl_seconds is not None:
            logger.warning("DynamoDbDocumentStore ignores ttl_seconds=%s on %s%s", ttl_seconds, collection, fields)
        if not unique:
            return
        self._table(collection, key_fields=list(fields))
        keyed_by = self._key_fields(collection)
        if keyed_by is None:
            raise NotImplementedError(
                f"DynamoDbDocumentStore cannot add the unique index {list(fields)!r} to {collection!r}: "
                "its table was created by a write before any unique index"
            )
        if keyed_by != list(fields):
            raise NotImplementedError(
                f"DynamoDbDocumentStore keys {collection!r} by {keyed_by!r}; "
                f"a second unique index {list(fields)!r} is unsupported"
            )
