"""Shared in-memory DocumentStore fake for unit tests.

Covers the Eq-keyed reads/inserts the store layer exercises, records
``ensure_index`` calls in ``indexes``, and simulates version races via
``fail_inserts`` (the first N inserts store the doc *and* raise
DuplicateKeyError, so a retry loop must re-read and advance). Operators no
current unit test needs raise NotImplementedError. Subclassing DocumentStore
keeps the fake honest: adding an abstract method breaks instantiation until the
fake implements it.
"""

from __future__ import annotations

import re
import asyncio
import base64
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.env_providers import EnvironmentGatewayProvider
from agent_env.providers.env_state import LocalPostgresStateProvider
from agent_env.store.base import ObjectAlreadyExistsError, ObjectNotFoundError
from agent_env.store.document_store import DocumentStore, DuplicateKeyError, Eq, Filter
from agent_env.store.image_store import ImageStore
from agent_env.store.object_store import DEFAULT_CONTENT_TYPE, LocalFilesystemObjectStore, ObjectMetadata, ObjectStore


class FakeDocumentStore(DocumentStore):
    def __init__(self, fail_inserts: int = 0) -> None:
        self.docs: list[dict] = []
        self.indexes: list[tuple[list[str], bool]] = []
        self.ttls = {}
        self._to_fail = fail_inserts

    def ensure_index(self, collection, fields, unique=False, ttl_seconds=None):
        self.indexes.append((fields, unique))
        if ttl_seconds is not None:
            self.ttls[(collection, tuple(fields))] = ttl_seconds

    @staticmethod
    def _match(doc: dict, filter: Filter) -> bool:
        for field, preds in filter.conditions.items():
            for p in preds:
                assert isinstance(p, Eq), "FakeDocumentStore supports only Eq predicates"
                if doc.get(field) != p.value:
                    return False
        return True

    def find_one(self, collection, filter, sort=None):
        matches = [d for d in self.docs if self._match(d, filter)]
        if sort and sort.keys:
            key = sort.keys[0]
            matches = sorted(matches, key=lambda d: d.get(key.field), reverse=key.descending)
        return matches[0] if matches else None

    def insert(self, collection, doc):
        self.docs.append(dict(doc))
        if self._to_fail > 0:
            self._to_fail -= 1
            raise DuplicateKeyError("simulated race")

    def query(self, *a, **k):
        raise NotImplementedError

    def count(self, *a, **k):
        raise NotImplementedError

    def update(self, collection, filter, update, upsert=False):
        for d in self.docs:
            if self._match(d, filter):
                d.update(update.set)
                for path in update.unset:
                    d.pop(path, None)
                return 1
        return 0

    def update_one_and_get(self, *a, **k):
        raise NotImplementedError

    def replace(self, *a, **k):
        raise NotImplementedError

    def delete(self, *a, **k):
        raise NotImplementedError


class FakeObjectStore(ObjectStore):
    """In-memory ObjectStore fake addressed by ``fake://{root}/{key}`` urls.

    Key-based ops target the configured home ``root``; ``_at`` ops address any
    root by url, so a single fake can prove the cross-root (cross-bucket) split
    the real backends have but that can't be unit-tested without network.
    """

    def __init__(self, root: str = "home") -> None:
        self._root = root
        self.objects: dict[str, bytes] = {}
        self._metadata: dict[str, ObjectMetadata] = {}

    def _record(self, url, data, content_type):
        self.objects[url] = data
        self._metadata[url] = ObjectMetadata(content_type=content_type, size=len(data), last_modified=datetime.now(timezone.utc))

    def put(self, key, data, content_type=DEFAULT_CONTENT_TYPE, allow_overwrite=False):
        url = self.object_url(key)
        if url in self.objects and not allow_overwrite:
            raise ObjectAlreadyExistsError(f"Object already exists at {url}.")
        self._record(url, bytes(data), content_type)
        return url

    def put_file(self, key, file_path, content_type=DEFAULT_CONTENT_TYPE):
        return self._put_file(self.object_url(key), file_path, content_type)

    def put_file_at(self, object_url, file_path, content_type=DEFAULT_CONTENT_TYPE):
        return self._put_file(object_url, file_path, content_type)

    def _put_file(self, url, file_path, content_type):
        if url in self.objects:
            raise ObjectAlreadyExistsError(f"Object already exists at {url}.")
        with open(file_path, "rb") as f:
            self._record(url, f.read(), content_type)
        return url

    def get(self, object_url):
        if object_url not in self.objects:
            raise ObjectNotFoundError(f"No object at {object_url}.")
        return self.objects[object_url]

    def download_to_file(self, object_url, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.get(object_url))

    def get_object_metadata(self, key):
        return self._metadata.get(self.object_url(key))

    def get_object_metadata_at(self, object_url):
        return self._metadata.get(object_url)

    def list(self, prefix):
        home = f"fake://{self._root}/"
        return [url[len(home):] for url in self.objects if url.startswith(home + prefix)]

    def list_at(self, url_prefix):
        return [url for url in self.objects if url.startswith(url_prefix)]

    def object_url(self, key):
        return f"fake://{self._root}/{key}"

    def get_object_key(self, object_url):
        home = f"fake://{self._root}/"
        if not object_url.startswith(home):
            raise ValueError(f"{object_url!r} is not an object in {home}.")
        return object_url[len(home):]


class RecordingObjectStore(FakeObjectStore):
    """A FakeObjectStore that records the keys each key-based read addressed.

    Lets a test assert *which* store served a read, not merely what it returned
    — the distinction that separates "read through the configured store" from
    "read through an ambient client that happened to answer the same way".
    """

    def __init__(self, objects: dict[str, bytes] | None = None, root: str = "home") -> None:
        super().__init__(root=root)
        self.exists_keys: list[str] = []
        self.read_keys: list[str] = []
        for key, data in (objects or {}).items():
            self.put(key, data)

    def exists(self, key):
        self.exists_keys.append(key)
        return super().exists(key)

    def read(self, key):
        self.read_keys.append(key)
        return super().read(key)


class SigningObjectStore(LocalFilesystemObjectStore):
    """A filesystem object store that signs urls, as a remote store does; the urls lead nowhere."""

    def signed_get_url(self, object_url, expires_in=3600):
        return f"https://objects.example.test/{self.get_object_key(object_url)}"

    signed_put_url = signed_get_url


class FakeImageStore(ImageStore):
    """In-memory ImageStore fake: records ensure_repository calls, no docker involved.

    Subclassing ImageStore keeps the fake honest — a new abstract method breaks
    instantiation until the fake implements it.
    """

    def __init__(self, registry: str = "fake.registry") -> None:
        self._registry = registry
        self.repositories: list[str] = []

    def image_ref(self, repository, tag):
        return f"{self._registry}/{repository}:{tag}"

    def ensure_repository(self, repository):
        self.repositories.append(repository)

    def auth(self, ref):
        return None


COLLECTED = {"/app/artifact/report.pdf": b"%PDF", "/app/artifact/sub/notes.md": b"# notes"}


class _Stream:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self):
        return self._data


class CollectingVm:
    """A VM serving ``COLLECTED`` to collect_artifacts' size probe and base64 read, and exiting 0 with ``ok``
    otherwise."""

    mode = "vm"

    async def exec_script(self, script: str) -> str:
        return ""

    async def exec_with_output(self, *args):
        if "stat" in args:
            return 0, str(len(COLLECTED[args[-1]])), ""
        return 0, "ok", ""

    async def exec(self, *args):
        path = args[-1].split("base64 < ", 1)[1].strip("'")
        return SimpleNamespace(stdout=_Stream(base64.b64encode(COLLECTED[path])), stderr=_Stream(b""),
                               wait=lambda: asyncio.sleep(0, result=0))


class SnapshotSandbox:
    """A VM whose servicedb answers the changelog count and the container lookup, and whose saved snapshot image
    answers the size and range reads that copy it off; records every script."""

    mode = "vm"
    tarball = b"snapshot image " * 1000

    def __init__(self) -> None:
        self.scripts: list[str] = []

    async def exec_script(self, script: str) -> str:
        self.scripts.append(script)
        if script.startswith("wc -c < "):
            return f"{len(self.tarball)}\n"
        if read := re.fullmatch(r"tail -c \+(\d+) \S+ \| head -c (\d+) \| base64", script):
            start, length = int(read[1]) - 1, int(read[2])
            return base64.b64encode(self.tarball[start:start + length]).decode()
        return "0\n" if "_changelog" in script else "container-1\n"


def reattach_for_snapshot(monkeypatch, env_id: str, env_version: int, universe_id: str,
                          universe_version: int) -> SnapshotSandbox:
    """Make ``EnvSnapshot.create`` find ``env_id`` deployed with ``universe_id`` loaded, and reconnect to a
    ``SnapshotSandbox``, which it returns."""
    deployed = DeployedGatewayEnv(env_id=env_id, env_version=env_version, sandbox_id="sb-1", gateway_url="https://gw",
                                  mcp_url="https://gw/mcp", db_web_url=None)
    instances = SimpleNamespace(get=lambda instance_id: deployed,
                                get_environment_universe=lambda instance_id: {"id": universe_id, "version": universe_version})
    provider = EnvironmentGatewayProvider()
    provider._state_provider = LocalPostgresStateProvider()
    reattached = SimpleNamespace(_sandbox=SnapshotSandbox(), _env_provider=provider)
    monkeypatch.setattr("agent_env.env.store.get_env_instance_store", lambda: instances)
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *args: MultiEnv(id=env_id, version=env_version, mcp_server_envs=[]))
    monkeypatch.setattr(MultiEnv, "from_deployed_env", classmethod(lambda cls, record: asyncio.sleep(0, result=reattached)))
    return reattached._sandbox
