"""Firestore with MongoDB compatibility as a DocumentStore (the ``gcp`` extra)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Optional, TypeVar

import google.auth
import pymongo
from google.api_core.exceptions import RetryError
from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError, TransportError
from pymongo import MongoClient, ReturnDocument
from pymongo.database import Database
from pymongo.auth_oidc import OIDCCallback, OIDCCallbackContext, OIDCCallbackResult
from pymongo.errors import PyMongoError

from agent_env.config.errors import ConfigError
from agent_env.store import _google
from agent_env.store.document_store.document_store import DuplicateKeyError, Filter, Sort, UpdateSpec
from agent_env.store.document_store.mongo_document_store import (
    MongoDocumentStore,
    _to_mongo_filter,
    _strip_id,
    _to_mongo_update,
    _unique_violations,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Stamped with a fresh value by every update and replace, and stripped from every read.
WRITE_STAMP_FIELD = "_agentenv_write"

# Firestore's MongoDB endpoint listens on the TLS port only.
_PORT = 443

# The client's timeouts, as MongoDocumentStore's; index builds get their own below.
_SERVER_SELECTION_TIMEOUT_MS = 30000
_CONNECT_TIMEOUT_MS = 15000
_SOCKET_TIMEOUT_MS = 30000

# createIndex answers only once the index is built, which takes a minute or more even on an
# empty collection; the client's 30 s socket timeout would end every first build.
_INDEX_BUILD_SECONDS = 600

# A transaction that loses to another writer on the same document is refused (code 112) and
# must be retried by the client.
_WRITE_CONFLICT = 112
_TRANSACTION_ATTEMPTS = 20
_TRANSACTION_BACKOFF_SECONDS = 0.02


# Index builds this process has finished, and a lock per index being built, shared by every store
# on one backend: Firestore answers a createIndex re-issued during a build before the index
# enforces anything, so only the first caller may issue it.
_built_indexes: set[tuple] = set()
_index_locks: dict[tuple, threading.Lock] = {}
_index_locks_guard = threading.Lock()


def _strip(doc: Optional[dict]) -> Optional[dict]:
    if doc is not None:
        doc.pop(WRITE_STAMP_FIELD, None)
    return doc


class _AccessTokenCallback(OIDCCallback):
    """MONGODB-OIDC over Application Default Credentials: Firestore takes a Google OAuth2
    access token, so this works wherever ADC does, not only on Google Cloud."""

    def __init__(self, credentials: Credentials) -> None:
        self._credentials = credentials
        self._request = _google.pooled_request()
        self._lock = threading.Lock()

    def fetch(self, context: OIDCCallbackContext) -> OIDCCallbackResult:
        with self._lock:
            if not self._credentials.valid:
                try:
                    _google.RETRY(self._credentials.refresh)(self._request)
                except RetryError as e:
                    raise TransportError(f"Google issued no access token within the retry window: {e.cause}") from e
            return OIDCCallbackResult(access_token=self._credentials.token)


class FirestoreMongoDocumentStore(MongoDocumentStore):
    """``MongoDocumentStore`` on a Firestore database with MongoDB compatibility.

    Firestore takes the MongoDB wire protocol, but a guarded write that loses a race to
    another writer on the same document is reported as matched: ``update_one`` counts it
    and ``find_one_and_update`` hands back the winner's document. So every update and
    replace here stamps ``WRITE_STAMP_FIELD`` with a fresh value, and counts as applied only
    when it modified the document or returned its own stamp. A pre-image read runs in a
    transaction instead, since a pre-image cannot carry the stamp. An upsert that loses an
    insert race is retried once, as MongoDB's server does for itself.

    Raw pymongo access to the same collections (``database``, ``mongo_collection``,
    ``Config.db``) bypasses all of this: readers see the stamp field, and a raw guarded write
    that loses a race is still reported as matched.
    """

    def __init__(self, database: Database, *, backend: Optional[str] = None) -> None:
        """``backend`` names the database, so that stores on it share their index builds;
        a store without one coordinates only its own callers."""
        super().__init__(database)
        self._backend = backend or uuid.uuid4().hex

    @classmethod
    def from_config(cls, *, host: str, database: str) -> FirestoreMongoDocumentStore:
        """Connect to the database at ``host`` (``<uid>.<location>.firestore.goog``) as the
        Application Default Credentials' identity, and ping it."""
        try:
            credentials, _ = google.auth.default(scopes=_google.SCOPES)
        except (DefaultCredentialsError, OSError) as e:
            raise ConfigError(f"No Google credentials for Firestore database {database!r}: {e}") from e
        client = MongoClient(
            f"mongodb://{host}:{_PORT}/{database}",
            loadBalanced=True,
            tls=True,
            authMechanism="MONGODB-OIDC",
            authMechanismProperties={"OIDC_CALLBACK": _AccessTokenCallback(credentials)},
            serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
            connectTimeoutMS=_CONNECT_TIMEOUT_MS,
            socketTimeoutMS=_SOCKET_TIMEOUT_MS,
            retryReads=True,
            retryWrites=False,  # Firestore rejects retryable writes' transaction numbers
        )
        try:
            client.admin.command("ping")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Firestore database {database!r}: {e}") from e
        logger.info("Connected to Firestore database: %s", database)
        return cls(client[database], backend=f"{host}/{database}")

    def find_one(self, collection: str, filter: Filter, sort: Optional[Sort] = None) -> Optional[dict]:
        return _strip(super().find_one(collection, filter, sort))

    def query(
        self,
        collection: str,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        return [_strip(doc) for doc in super().query(collection, filter, sort, limit, offset)]

    def update(self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool = False) -> int:
        spec, _ = self._stamped(update)
        result = self._upserting(
            lambda: self._c(collection).update_one(_to_mongo_filter(filter), spec, upsert=upsert), upsert
        )
        return 1 if result.upserted_id is not None else result.modified_count

    def update_one_and_get(
        self,
        collection: str,
        filter: Filter,
        update: UpdateSpec,
        return_after: bool = True,
        upsert: bool = False,
    ) -> Optional[dict]:
        if not return_after:
            return self._update_returning_pre_image(collection, filter, update, upsert)
        spec, stamp = self._stamped(update)
        doc = self._upserting(
            lambda: self._c(collection).find_one_and_update(
                _to_mongo_filter(filter), spec, return_document=ReturnDocument.AFTER, upsert=upsert
            ),
            upsert,
        )
        if doc is None or doc.get(WRITE_STAMP_FIELD) != stamp:
            return None
        return _strip(_strip_id(doc))

    def replace(self, collection: str, filter: Filter, doc: dict, upsert: bool = False) -> int:
        body = {**doc, WRITE_STAMP_FIELD: uuid.uuid4().hex}
        result = self._upserting(
            lambda: self._c(collection).replace_one(_to_mongo_filter(filter), body, upsert=upsert), upsert
        )
        return 1 if result.upserted_id is not None else result.modified_count

    def ensure_index(
        self,
        collection: str,
        fields: list[str],
        unique: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        key = (self._backend, collection, tuple(fields), unique, ttl_seconds)
        with _index_locks_guard:
            lock = _index_locks.setdefault(key, threading.Lock())
        with lock:
            if key in _built_indexes:
                return
            with pymongo.timeout(_INDEX_BUILD_SECONDS):
                self._create_index(collection, fields, unique=unique, ttl_seconds=ttl_seconds)
            _built_indexes.add(key)

    def _stamped(self, update: UpdateSpec) -> tuple[dict, str]:
        stamp = uuid.uuid4().hex
        spec = _to_mongo_update(update)
        spec.setdefault("$set", {})[WRITE_STAMP_FIELD] = stamp
        return spec, stamp

    def _upserting(self, write: Callable[[], T], upsert: bool) -> T:
        try:
            with _unique_violations():
                return write()
        except DuplicateKeyError:
            if not upsert:
                raise
        # Two upserts both missed and the other inserted first: this one now matches it.
        with _unique_violations():
            return write()

    def _update_returning_pre_image(
        self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool
    ) -> Optional[dict]:
        mongo_filter, spec = _to_mongo_filter(filter), self._stamped(update)[0]
        coll = self._c(collection)
        for attempt in range(_TRANSACTION_ATTEMPTS):
            try:
                with _unique_violations(), self._db.client.start_session() as session:
                    with session.start_transaction():
                        before = coll.find_one(mongo_filter, session=session)
                        if before is not None:
                            coll.update_one({"_id": before["_id"]}, spec, session=session)
                        elif upsert:
                            coll.update_one(mongo_filter, spec, upsert=True, session=session)
                return _strip(_strip_id(before))
            except PyMongoError as e:
                if getattr(e, "code", None) != _WRITE_CONFLICT and not e.has_error_label("TransientTransactionError"):
                    raise
                time.sleep(_TRANSACTION_BACKOFF_SECONDS * (attempt + 1))
        raise RuntimeError(f"a pre-image update on {collection!r} lost {_TRANSACTION_ATTEMPTS} write conflicts in a row")
