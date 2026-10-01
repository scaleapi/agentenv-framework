"""Namespace-addressed stores: an ``@local/…`` id lives in the ``@local`` namespace's local stores,
every other id in the configured ones.

Routing runs only once a process turns it on (the CLI does, for the length of a command). Everywhere
else each getter returns the configured store exactly as before, so a service keeps rejecting every
``@`` id. The ``@local`` namespace keeps its documents in a file of its own beside the default local
store's, so each file only ever holds one namespace's documents; objects and images share the
per-user object store and local registry, whose urls and repositories say whose they are.

Entity documents route by their id. What a task run records (instances, journals, state, runs,
reviews, conversations, snapshots) and the objects it writes under keys of its own follow the run
scope ``Task.run()`` sets from its task id. A read that neither rule places reads both stores, the
reader's own namespace first: an ``@local`` run sees its records before the registry's, and anyone
else sees the registry's first. While an ``@local`` task runs, a write that would reach a configured
store is refused (the write fence), and a bare entity may never name an ``@local`` id (references
point outward only).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Optional, TypeVar

from agent_env.config.errors import ConfigError
from agent_env.store.document_store.document_store import (
    DocumentStore,
    Eq,
    Filter,
    In,
    Sort,
    UpdateSpec,
    _dig,
    is_entity_collection,
)
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore
from agent_env.store.ids import LOCAL_NAMESPACE, LOCAL_PREFIX, is_local_id, parse_namespace, validate_local_id
from agent_env.store.image_store import ImageStore, OciRegistryImageStore
from agent_env.store.image_store.oci_registry_credentials import (
    RegistryAuth,
    normalize_registry_host,
    registry_host_from_ref,
)
from agent_env.store.object_store import ObjectStore
from agent_env.store.object_store.object_store import (
    DEFAULT_CONTENT_TYPE,
    HttpGetGrant,
    HttpPutGrant,
    ObjectMetadata,
    UploadPolicy,
)

# The fields a run-time document names its owner by: one holding an @local id places it locally.
_OWNER_FIELDS = ("id", "instance_id", "task_id", "task_instance_id", "env_id", "agent_id")
_LOCAL_REPOSITORY_PREFIX = f"{LOCAL_NAMESPACE}/"
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_T = TypeVar("_T")

_enabled = False
_scopes = 0
_scopes_lock = threading.Lock()
_run_namespace: ContextVar[Optional[str]] = ContextVar("agent_env_run_namespace", default=None)


class LocalRunWriteError(ValueError):
    """A write an ``@local`` task run may not make, since it would land in a configured store."""


class OutwardReferenceError(ValueError):
    """A write that would point outside the ``@local`` namespace's stores at an ``@local`` id."""


def enable_namespace_routing() -> None:
    """Route ``@local`` ids to the per-user local stores in this process. The CLI turns this on; a
    service never does, so its stores stay exactly the configured ones."""
    global _enabled
    _enabled = True


def disable_namespace_routing() -> None:
    global _enabled
    _enabled = False


def namespace_routing_enabled() -> bool:
    return _enabled or _scopes > 0


@contextmanager
def namespace_routing() -> Iterator[None]:
    """Route namespaces for the length of the block. Overlapping blocks (two commands run in one
    process) keep routing on until the last of them ends."""
    global _scopes
    with _scopes_lock:
        _scopes += 1
    try:
        yield
    finally:
        with _scopes_lock:
            _scopes -= 1


@contextmanager
def run_scope(owner_id: str) -> Iterator[None]:
    """The run in progress, a task's or a deployment the CLI makes: what it records follows its owner id's
    namespace. A run started inside an ``@local`` run stays local, so a nested registry task can't reopen
    the configured stores."""
    outer = _run_namespace.get()
    token = _run_namespace.set(LOCAL_NAMESPACE if outer == LOCAL_NAMESPACE else parse_namespace(owner_id) or "")
    try:
        yield
    finally:
        _run_namespace.reset(token)


def in_local_run() -> bool:
    return namespace_routing_enabled() and _run_namespace.get() == LOCAL_NAMESPACE


def refuse_in_local_run(what: str) -> None:
    if in_local_run():
        raise LocalRunWriteError(
            f"an @local task run writes only to the local stores, and {what} would reach a configured store"
        )


def refuse_local_derivation(entity_id: str, kind: str, doing: str) -> None:
    """Refuse ``doing`` to an ``@local`` entity whose tasks, artifacts or objects are named after it
    in a form that doesn't keep the ``@local`` namespace yet."""
    if is_local_id(entity_id):
        raise ValueError(f"{entity_id!r} is an @local {kind}, and {doing} one isn't supported yet")


def refuse_local_references(entity_id: object, value: Any) -> None:
    """Refuse a bare entity naming an ``@local`` id anywhere in ``value``: a shared store must
    never point into one person's local stores."""
    path = _local_reference(value, "")
    if path is not None:
        raise OutwardReferenceError(
            f"{entity_id!r} is not an @local id, so it can't reference one: {path or 'its value'} names an @local id"
        )


def _local_reference(value: Any, path: str) -> Optional[str]:
    if isinstance(value, str):
        return path if is_local_id(value) else None
    if isinstance(value, Mapping):
        items = ((f"{path}.{key}" if path else str(key), item) for key, item in value.items())
    elif isinstance(value, (list, tuple, set)):
        items = ((f"{path}[{index}]", item) for index, item in enumerate(value))
    else:
        return None
    return next((found for item_path, item in items if (found := _local_reference(item, item_path)) is not None), None)


def _update_values(update: UpdateSpec) -> dict:
    return {"set": update.set, "add_to_set": update.add_to_set, "push": update.push}


def _merge_sort(docs: list[dict], sort: Optional[Sort], *, absent_last: bool = False) -> list[dict]:
    """Order documents from both stores the way each store orders its own: an absent or null value
    first ascending and last descending (``query``, ``find_one``), or last either way with
    ``absent_last`` (``latest_per_id``); a datetime (Mongo's) and an ISO timestamp string (SQLite's)
    compare as times."""
    if not sort or not sort.keys:
        return docs
    ordered = list(docs)
    for key in reversed(sort.keys):
        values = {id(doc): _dig(doc, key.field) for doc in ordered}
        present = [doc for doc in ordered if values[id(doc)][0] and values[id(doc)][1] is not None]
        absent = [doc for doc in ordered if not (values[id(doc)][0] and values[id(doc)][1] is not None)]
        present.sort(key=lambda doc: _comparable(values[id(doc)][1]), reverse=key.descending)
        ordered = present + absent if absent_last or key.descending else absent + present
    return ordered


def _comparable(value: Any) -> tuple:
    if isinstance(value, str) and _TIMESTAMP.match(value):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return ("str", value)
    if isinstance(value, datetime):
        return ("time", value if value.tzinfo else value.replace(tzinfo=timezone.utc))
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, (dict, list)):
        return ("json", json.dumps(value, sort_keys=True, default=str))
    return (type(value).__name__, value)


def _ids_split_by_namespace(collection: str, id_field: str) -> bool:
    """Whether no id can be in both stores: an entity's id decides which one it is written to."""
    return is_entity_collection(collection) and id_field == "id"


def _newer(version: Any, than: Any) -> bool:
    """Whether ``version`` is a number above ``than``; anything but a number is older than any number."""
    def number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    return number(version) and (not number(than) or version > than)


def _window(docs: list[dict], limit: Optional[int], offset: Optional[int]) -> list[dict]:
    docs = docs[offset or 0:]
    return docs[:limit] if limit else docs


def _filter_owner(filter: Filter) -> dict:
    """The field values an all-equality filter pins, the document an upsert would create."""
    return {path: preds[0].value for path, preds in filter.conditions.items() if len(preds) == 1 and isinstance(preds[0], Eq)}


def _filter_is_local(filter: Filter) -> Optional[bool]:
    """Whether the ids a filter pins are all ``@local`` (True) or all bare (False); None when it pins
    none, or both kinds."""
    values: list[Any] = []
    for pred in filter.conditions.get("id", ()):
        if isinstance(pred, Eq):
            values.append(pred.value)
        elif isinstance(pred, In):
            values.extend(pred.values)
    kinds = {is_local_id(value) for value in values if isinstance(value, str)}
    return kinds.pop() if len(kinds) == 1 else None


class LocalNamespaceDocumentStore(LocalSqliteDocumentStore):
    """The ``@local`` namespace's documents: its entities, and what its runs record."""

    def check_id(self, entity_id: str) -> None:
        if not is_local_id(entity_id):
            raise ValueError(f"id {entity_id!r} isn't an @local id, and the @local namespace's store holds only those")
        validate_local_id(entity_id)


class RoutingDocumentStore(DocumentStore):
    """The configured document store, with ``@local`` documents and ``@local`` runs' records kept in
    the ``@local`` namespace's store. Reads never create that store; writes open it with its indexes."""

    def __init__(self, configured: DocumentStore, local: DocumentStore, local_path: Path) -> None:
        self.configured = configured
        self.local = local
        self._local_path = local_path
        self._lock = threading.Lock()
        self._pending_local_indexes: Optional[list[tuple]] = []

    def find_one(self, collection: str, filter: Filter, sort: Optional[Sort] = None) -> Optional[dict]:
        found = [
            doc for store in self._readers(collection, filter)
            if (doc := self._read(store, lambda: store.find_one(collection, filter, sort))) is not None
        ]
        return _merge_sort(found, sort)[0] if found else None

    def query(
        self,
        collection: str,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        readers = self._readers(collection, filter)
        if len(readers) == 1:
            return self._read(readers[0], lambda: readers[0].query(collection, filter, sort, limit, offset))
        window = (offset or 0) + limit if limit else None
        merged = [doc for store in readers for doc in self._read(store, lambda: store.query(collection, filter, sort, window))]
        return _window(_merge_sort(merged, sort), limit, offset)

    def count(self, collection: str, filter: Filter) -> int:
        return sum(self._read(store, lambda: store.count(collection, filter)) for store in self._readers(collection, filter))

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
        def latest(store: DocumentStore, **page: Any) -> list[dict]:
            return store.latest_per_id(collection, filter, id_field=id_field, version_field=version_field, sort=sort, **page)

        readers = self._readers(collection, filter)
        if len(readers) == 1:
            return self._read(readers[0], lambda: latest(readers[0], limit=limit, offset=offset))
        if _ids_split_by_namespace(collection, id_field):
            window = (offset or 0) + limit if limit else None
            merged = [doc for store in readers for doc in self._read(store, lambda: latest(store, limit=window))]
        else:
            merged = self._latest_across(readers, collection, filter, id_field, version_field)
        return _window(_merge_sort(merged, sort, absent_last=True), limit, offset)

    def count_distinct(self, collection: str, filter: Filter, *, id_field: str = "id") -> int:
        readers = self._readers(collection, filter)
        if len(readers) == 1 or _ids_split_by_namespace(collection, id_field):
            return sum(
                self._read(store, lambda: store.count_distinct(collection, filter, id_field=id_field)) for store in readers
            )
        return len(self._latest_across(readers, collection, filter, id_field, "version"))

    def _latest_across(
        self, readers: list[DocumentStore], collection: str, filter: Filter, id_field: str, version_field: str,
    ) -> list[dict]:
        """Each id's latest document across both stores, where one id can be in both: the higher
        version wins, and a tie goes to the reader's own namespace."""
        latest: dict[str, dict] = {}
        for store in readers:
            docs = self._read(
                store, lambda: store.latest_per_id(collection, filter, id_field=id_field, version_field=version_field)
            )
            for doc in docs:
                key = json.dumps(_dig(doc, id_field)[1], sort_keys=True, default=str)
                held = latest.get(key)
                if held is None or _newer(_dig(doc, version_field)[1], _dig(held, version_field)[1]):
                    latest[key] = doc
        return list(latest.values())

    def insert(self, collection: str, doc: dict) -> None:
        store = self._writer(collection, doc, doc)
        self._read(store, lambda: store.insert(collection, doc))

    def update(self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool = False) -> int:
        if upsert:
            store = self._writer(collection, _filter_owner(filter), _update_values(update))
            return self._read(store, lambda: store.update(collection, filter, update, upsert=True))
        for store in self._updaters(collection, filter, _update_values(update)):
            if matched := self._read(store, lambda: store.update(collection, filter, update)):
                return matched
        return 0

    def update_one_and_get(
        self,
        collection: str,
        filter: Filter,
        update: UpdateSpec,
        return_after: bool = True,
        upsert: bool = False,
    ) -> Optional[dict]:
        if upsert:
            store = self._writer(collection, _filter_owner(filter), _update_values(update))
            return self._read(store, lambda: store.update_one_and_get(collection, filter, update, return_after, upsert=True))
        for store in self._updaters(collection, filter, _update_values(update)):
            if (doc := self._read(store, lambda: store.update_one_and_get(collection, filter, update, return_after))) is not None:
                return doc
        return None

    def replace(self, collection: str, filter: Filter, doc: dict, upsert: bool = False) -> int:
        if upsert:
            store = self._writer(collection, {**_filter_owner(filter), **doc}, doc)
            return self._read(store, lambda: store.replace(collection, filter, doc, upsert=True))
        for store in self._updaters(collection, filter, doc):
            if matched := self._read(store, lambda: store.replace(collection, filter, doc)):
                return matched
        return 0

    def delete(self, collection: str, filter: Filter) -> int:
        for store in self._updaters(collection, filter, None):
            if deleted := self._read(store, lambda: store.delete(collection, filter)):
                return deleted
        return 0

    def ensure_index(
        self,
        collection: str,
        fields: list[str],
        unique: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.configured.ensure_index(collection, fields, unique, ttl_seconds)
        with self._lock:
            if self._pending_local_indexes is not None:
                self._pending_local_indexes.append((collection, fields, unique, ttl_seconds))
                return
        self.local.ensure_index(collection, fields, unique, ttl_seconds)

    def check_id(self, entity_id: str) -> None:
        if is_local_id(entity_id):
            self.local.check_id(entity_id)
            return
        if LOCAL_PREFIX in entity_id:
            raise ValueError(
                f"id {entity_id!r} contains an @local id but isn't one: an id derived from an @local id "
                "has to keep the @local namespace"
            )
        self.configured.check_id(entity_id)

    def _placed(self, collection: str, filter: Filter) -> Optional[DocumentStore]:
        if not is_entity_collection(collection):
            return None
        local = _filter_is_local(filter)
        return None if local is None else self.local if local else self.configured

    def _readers(self, collection: str, filter: Filter) -> list[DocumentStore]:
        """The stores a read consults, the reader's own namespace first: a key can be in both, when
        an ``@local`` run and a registry run each recorded it."""
        placed = self._placed(collection, filter)
        if placed is self.configured:
            return [self.configured]
        local = [self.local] if self._local_path.exists() else []
        if placed is self.local:
            return local
        return [*local, self.configured] if in_local_run() else [self.configured, *local]

    def _read(self, store: DocumentStore, read: Callable[[], _T]) -> _T:
        if store is not self.local:
            return read()
        try:
            return read()
        except (sqlite3.DatabaseError, OSError) as e:
            raise ConfigError(
                f"the @local store at {self._local_path} can't be read ({e}); move it aside to start a new one"
            ) from e

    def _writer(self, collection: str, owner: Mapping, payload: Any) -> DocumentStore:
        if is_entity_collection(collection):
            if isinstance(owner.get("id"), str):
                self.check_id(owner["id"])
            local = isinstance(owner.get("id"), str) and is_local_id(owner["id"])
        else:
            local = in_local_run() or any(isinstance(owner.get(f), str) and is_local_id(owner[f]) for f in _OWNER_FIELDS)
        if local:
            return self._open_local()
        self._check_configured_write(collection, owner.get("id"), payload)
        return self.configured

    def _updaters(self, collection: str, filter: Filter, payload: Any) -> Iterator[DocumentStore]:
        """The stores an update of ``filter``'s match tries, in order: the one its id places it in,
        else the caller's own namespace's first. Inside an ``@local`` run only the local store is
        written, and a match that lives only in the configured store is refused rather than missed."""
        placed = self._placed(collection, filter)
        local = placed is not self.configured and self._local_path.exists()
        if placed is None and not in_local_run():
            self._check_configured_write(collection, _filter_owner(filter).get("id"), payload)
            yield self.configured
            if local:
                yield self._open_local()
            return
        if local:
            yield self._open_local()
        if placed is self.configured:
            self._check_configured_write(collection, _filter_owner(filter).get("id"), payload)
            yield self.configured
        elif placed is None and self.configured.find_one(collection, filter) is not None:
            refuse_in_local_run(f"updating {collection!r} (its record is in the configured store)")

    def _check_configured_write(self, collection: str, entity_id: object, payload: Any) -> None:
        refuse_in_local_run(f"this write to {collection!r}")
        if is_entity_collection(collection) and payload is not None:
            refuse_local_references(entity_id, payload)

    def _open_local(self) -> DocumentStore:
        with self._lock:
            for spec in self._pending_local_indexes or ():
                self._read(self.local, lambda: self.local.ensure_index(*spec))
            self._pending_local_indexes = None
        return self.local


def object_store_holding(object_url: str, configured: ObjectStore, local: ObjectStore) -> ObjectStore:
    """The store an object_url addresses: the local store's own urls are local unless the configured
    store owns them too, and any other url is the configured store's, as it was before routing."""
    return local if local.owns(object_url) and not configured.owns(object_url) else configured


def configured_store(store: Any) -> Any:
    """The configured store behind ``store``: a routing store's or view's configured side, else
    ``store`` itself. For reports that name the backend a user configured."""
    return store.configured if isinstance(store, (RoutingDocumentStore, LocalRunObjectStore, LocalRunImageStore)) else store


def image_store_holding(ref: str, configured: ImageStore, local: ImageStore) -> ImageStore:
    host = registry_host_from_ref(ref)
    if isinstance(local, OciRegistryImageStore) and host == normalize_registry_host(local.registry_host):
        return local
    return configured


class LocalRunObjectStore(ObjectStore):
    """The object store an ``@local`` task run sees: keys go to the local store, an object_url to the
    store it addresses, and a write that would reach the configured store is refused."""

    def __init__(self, configured: ObjectStore, local: ObjectStore) -> None:
        self.configured = configured
        self.local = local

    @property
    def supports_transfer_grants(self) -> bool:
        return self.local.supports_transfer_grants

    def grants_reach(self, sandbox_type: str | None) -> bool:
        # A run writes to the local store, so its grants must reach the agent too.
        return self.local.grants_reach(sandbox_type)

    @property
    def max_single_upload_bytes(self) -> int | None:
        return self.local.max_single_upload_bytes

    def put(self, key: str, data: bytes, content_type: str = DEFAULT_CONTENT_TYPE, allow_overwrite: bool = False) -> str:
        return self.local.put(key, data, content_type, allow_overwrite)

    def put_file(self, key: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self.local.put_file(key, file_path, content_type)

    def put_file_at(self, object_url: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self._writing(object_url).put_file_at(object_url, file_path, content_type)

    def get(self, object_url: str) -> bytes:
        return self._at(object_url).get(object_url)

    def exists(self, key: str) -> bool:
        return self.local.exists(key)

    def get_object_metadata(self, key: str) -> ObjectMetadata | None:
        return self.local.get_object_metadata(key)

    def get_object_metadata_at(self, object_url: str) -> ObjectMetadata | None:
        return self._at(object_url).get_object_metadata_at(object_url)

    def list(self, prefix: str) -> list[str]:
        return self.local.list(prefix)

    def list_at(self, url_prefix: str) -> list[str]:
        return self._at(url_prefix).list_at(url_prefix)

    def object_url(self, key: str) -> str:
        return self.local.object_url(key)

    def get_object_key(self, object_url: str) -> str:
        return self._at(object_url).get_object_key(object_url)

    def owns(self, object_url: str) -> bool:
        return self.local.owns(object_url) or self.configured.owns(object_url)

    def download_to_file(self, object_url: str, dest_path: str) -> None:
        self._at(object_url).download_to_file(object_url, dest_path)

    def open(self, object_url: str) -> BinaryIO:
        return self._at(object_url).open(object_url)

    def read(self, key: str) -> bytes:
        return self.local.read(key)

    def signed_get_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        return self._at(object_url).signed_get_url(object_url, expires_in)

    def signed_put_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        return self._writing(object_url).signed_put_url(object_url, expires_in)

    def signed_post(self, url_prefix: str, *, expires_in: int = 3600, max_bytes: int | None = None) -> dict | None:
        return self._writing(url_prefix).signed_post(url_prefix, expires_in=expires_in, max_bytes=max_bytes)

    def shared_credentials_env(self) -> dict[str, str]:
        return self.configured.shared_credentials_env()

    def issue_read_grant(self, object_url: str, *, expires_in: int = 3600) -> HttpGetGrant:
        return self._at(object_url).issue_read_grant(object_url, expires_in=expires_in)

    def issue_write_grant(self, object_url: str, *, media_type: str, max_bytes: int, expires_in: int = 3600) -> HttpPutGrant:
        return self._writing(object_url).issue_write_grant(
            object_url, media_type=media_type, max_bytes=max_bytes, expires_in=expires_in
        )

    def issue_upload_policy(self, prefix_url: str, *, max_object_bytes: int, expires_in: int) -> UploadPolicy:
        return self._writing(prefix_url).issue_upload_policy(prefix_url, max_object_bytes=max_object_bytes, expires_in=expires_in)

    def _at(self, object_url: str) -> ObjectStore:
        return object_store_holding(object_url, self.configured, self.local)

    def _writing(self, object_url: str) -> ObjectStore:
        store = self._at(object_url)
        if store is self.configured:
            refuse_in_local_run(f"a write to {object_url!r}")
        return store


class LocalRunImageStore(ImageStore):
    """The image store an ``@local`` task run sees: ``@local`` repositories are the local registry's,
    pull auth comes from the store a ref names, and creating a configured repository is refused."""

    def __init__(self, configured: ImageStore, local: ImageStore) -> None:
        self.configured = configured
        self.local = local

    def image_ref(self, repository: str, tag: str) -> str:
        return self._for(repository).image_ref(repository, tag)

    def ensure_repository(self, repository: str) -> None:
        store = self._for(repository)
        if store is self.configured:
            refuse_in_local_run(f"creating repository {repository!r}")
        store.ensure_repository(repository)

    def owns(self, ref: str) -> bool:
        return image_store_holding(ref, self.configured, self.local).owns(ref)

    def auth(self, ref: str) -> RegistryAuth | None:
        return image_store_holding(ref, self.configured, self.local).auth(ref)

    def _for(self, repository: str) -> ImageStore:
        return self.local if repository.startswith(_LOCAL_REPOSITORY_PREFIX) else self.configured
