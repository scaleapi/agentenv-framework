"""The bundle ledger: what each version a bundle wrote was made from, so a re-run writes only what changed.

A row per written version records a digest of the write's inputs and one digest per input, so a new
version can say why it was written ("files changed: server.py"). The ledger lives in the ``@local``
namespace's own store, beside the entities it describes, and is never read through the router, so it
never reaches a configured store. Reading it never creates the store.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

from agent_env.a2a_agent.store import A2A_AGENTS_COLLECTION
from agent_env.artifact.registry import canonical_type, get_artifact_registry
from agent_env.artifact.store import ARTIFACTS_COLLECTION
from agent_env.config import get_config
from agent_env.env.registry import get_env_registry
from agent_env.env.store import ENVS_COLLECTION
from agent_env.eval.store import EVALS_COLLECTION
from agent_env.store import Filter, Sort
from agent_env.store.document_store import LocalSqliteDocumentStore
from agent_env.store.local_state import holding_locks
from agent_env.task.store import TASKS_COLLECTION

from .authoring import build_context_files, entry_files
from .parse import Bundle, BundleKind
from .plan import Plan, Write, env_writer, folder_walk, unpinned_store_refs
from .resolve import BuiltImage

LEDGER_COLLECTION = "bundle_ledger"
# Bumped by hand when what a digest covers changes, or a writer's output changes enough that every
# version it wrote should be written again.
SCHEME = 1

_COLLECTIONS = {"env": ENVS_COLLECTION, "agent": A2A_AGENTS_COLLECTION, "artifact": ARTIFACTS_COLLECTION,
                "task": TASKS_COLLECTION, "eval": EVALS_COLLECTION}
_LISTED = 5


@dataclass(frozen=True)
class Digest:
    value: str
    inputs: dict[str, Any]  # type, config, files {key: sha256}, needs and store_refs {"<kind> <id>": version}


@dataclass(frozen=True)
class Check:
    """Whether a write can reuse the version the bundle last wrote, and if not, why not."""

    write: Write
    digest: Digest | None  # None when the write's inputs aren't tracked, so it is always written
    version: int | None  # the version to reuse; None when the write needs a new one
    reasons: tuple[str, ...]
    stored: int | None  # the store's latest version of the id when checked
    needs: Mapping[tuple[str, str], int]  # the version hashed for each earlier write it needs, by (store, id)
    adopted: bool = False  # ``version`` was written by an interrupted run of this bundle, which didn't record it

    @property
    def unchanged(self) -> bool:
        return self.version is not None

    @property
    def next_version(self) -> int:
        """The version a new write lands on, unless another run writes the id first."""
        return (self.stored or 0) + 1


class Ledger:
    """One plan's rows in the ``@local`` namespace's store."""

    def __init__(self, store: LocalSqliteDocumentStore, plan: Plan):
        self._store = store
        self._plan = plan
        self._bundle = plan.bundle.bundle.id_root

    @classmethod
    def for_plan(cls, plan: Plan) -> Ledger:
        return cls(get_config().local_namespace_document_store(), plan)

    def check(self, write: Write, needs: Mapping[tuple[str, str], int]) -> Check:
        """Compare ``write`` with the latest version the ledger records for it. ``needs`` holds, by (store, id),
        the version each earlier write it needs left or, in a dry run, would leave; an env or agent is hashed with
        them, so one written anew rewrites it too."""
        stored = self._stored_version(write)
        digest = self.digest(write, needs)
        if digest is None:
            untracked = ("its inputs aren't tracked yet, so it is written every run",)
            return Check(write, None, None, untracked, stored, needs)
        orphan = self._orphan(write, stored, digest)
        if orphan is not None:
            return Check(write, digest, orphan, ("written by an interrupted run that didn't record it",), stored, needs,
                         adopted=True)
        row = self._latest(write.kind.store, write.id)
        recorded = row["version"] if row else None
        reasons = []
        if row is not None and row["bundle"] != self._bundle:
            reasons.append(f"last written by the bundle {row['bundle']}")
        if stored is None:
            reasons.append("new" if row is None else f"its v{recorded} is not in the store")
        elif stored != recorded:
            reasons.append(f"the store's latest, v{stored}, wasn't recorded by this bundle")
        if row is not None:
            reasons.extend(_changes(row, digest))
        return Check(write, digest, None if reasons else recorded, tuple(reasons), stored, needs)

    def digest(self, write: Write, needs: Mapping[tuple[str, str], int]) -> Digest | None:
        """What ``write`` is made from, given the version of each write it needs, or None when the ledger can't
        tell it all, so it is written every run."""
        if not _tracked(write):
            return None
        config = {"dockerfile": write.source.dockerfile} if isinstance(write.source, BuiltImage) else write.source.config
        inputs = {"type": _type(write), "config": _sha256(_canonical(config)),
                  "files": {}, "needs": {}, "store_refs": {}}
        if write.kind is BundleKind.ARTIFACT:
            built = isinstance(write.source, BuiltImage)
            for key, path in (build_context_files if built else entry_files)(self._plan.bundle.bundle,
                                                                              write.source.entry).items():
                inputs["files"][key] = _file_sha256(path) + (_mode(path) if built else "")
        elif write.kind in (BundleKind.ENV, BundleKind.AGENT):
            # An env's or agent's document records the versions of what it references, so one written anew means
            # it must be written again. A task or eval names its references without a version.
            for store, id in write.needs:
                inputs["needs"][f"{store} {id}"] = str(needs[store, id])
        if write.kind not in (BundleKind.TASK, BundleKind.EVAL):
            for (kind, id), version in unpinned_store_refs(self._plan, write).items():
                inputs["store_refs"][f"{kind} {id}"] = str(version)
        value = _sha256(_canonical({"scheme": SCHEME, "store": write.kind.store, "id": write.id, "inputs": inputs}))
        return Digest(value, inputs)

    def record(self, check: Check, write: Callable[[], int]) -> int:
        """Write ``check``'s entity with ``write``, which returns the version it wrote, and record it.
        A pending row marks the attempt until it's done, with its version once ``write`` returns. One an
        interrupted run of this bundle left is kept by ``check`` when its version can be, and dropped here."""
        key = {"store": check.write.kind.store, "id": check.write.id, "bundle": self._bundle}
        self._store.ensure_index(LEDGER_COLLECTION, ["store", "id", "status"])
        while self._store.delete(LEDGER_COLLECTION, Filter.of(**key, status="pending")):
            pass
        pending = {**key, "status": "pending", "scheme": SCHEME,
                   "digest": check.digest.value if check.digest else None,
                   "inputs": check.digest.inputs if check.digest else None, "at": _now()}
        self._store.insert(LEDGER_COLLECTION, pending)
        version = write()
        # Stamped before the check below, which can take seconds, so a run interrupted during it leaves the version
        # this attempt wrote, and the next run keeps it rather than writing it again.
        self._store.replace(LEDGER_COLLECTION, Filter.of(**key, status="pending"), {**pending, "version": version})
        done = {**pending, "status": "done", "version": version, "at": _now()}
        if check.digest is not None and self.digest(check.write, check.needs) != check.digest:
            # A file changed during the write, so what the version holds is unknown: the next run writes it again.
            done.update(digest=None, inputs=None)
        self._store.replace(LEDGER_COLLECTION, Filter.of(**key, status="pending"), done, upsert=True)
        return version

    def adopt(self, check: Check) -> None:
        """Record the version an interrupted run wrote, which ``check`` keeps, as that run would have."""
        key = {"store": check.write.kind.store, "id": check.write.id, "bundle": self._bundle}
        done = {**key, "status": "done", "scheme": SCHEME, "digest": check.digest.value, "inputs": check.digest.inputs,
                "version": check.version, "at": _now()}
        self._store.replace(LEDGER_COLLECTION, Filter.of(**key, status="pending"), done, upsert=True)

    def _orphan(self, write: Write, stored: int | None, digest: Digest) -> int | None:
        """The version an interrupted run of this bundle wrote and didn't record, when it's still the store's
        latest and was made from what ``write`` is now made from."""
        pending = self._find(LEDGER_COLLECTION, Filter.of(store=write.kind.store, id=write.id, bundle=self._bundle,
                                                          status="pending"))
        if pending is None or pending.get("version") is None or pending["version"] != stored:
            return None
        return stored if (pending["scheme"], pending["digest"]) == (SCHEME, digest.value) else None

    def _latest(self, store: str, id: str) -> dict | None:
        return self._find(LEDGER_COLLECTION, Filter.of(store=store, id=id, status="done"))

    def _stored_version(self, write: Write) -> int | None:
        doc = self._find(_COLLECTIONS[write.kind.store], Filter.of(id=write.id))
        return doc["version"] if doc else None

    def _find(self, collection: str, filter: Filter) -> dict | None:
        """The highest version matching ``filter``, read without creating the store."""
        if not self._store.path.exists():
            return None
        return self._store.find_one(collection, filter, sort=Sort.by("version"))


@contextmanager
def materializing(bundle: Bundle, on_wait: Callable[[], None] | None = None) -> Iterator[None]:
    """Hold a lock on every id the bundle's entries write while its entities are written. Another run that
    writes any of those ids, of this bundle or another, waits here, calling ``on_wait`` first. Task runs don't
    hold them."""
    with holding_locks((entry.id for entry in bundle.entries), on_wait):
        yield


def _tracked(write: Write) -> bool:
    """Whether the ledger can list everything ``write`` is made from. Not yet for a skill, nor for a type with
    a ``from_toml`` of its own, which may read its folder in ways it can't see. An agent or env is made from its
    toml alone: its images and envs are references. A built image is made from its build context,
    ``build_context_files``; what the build fetches (its base image, packages) isn't an input."""
    if isinstance(write.source, BuiltImage):
        return True
    if write.kind is BundleKind.SKILL:
        return False
    if write.kind is BundleKind.ARTIFACT:
        return folder_walk(get_artifact_registry().get(_type(write))) is not None
    if write.kind is BundleKind.ENV:
        return env_writer(get_env_registry().get(_type(write)))
    return True


def _type(write: Write) -> str:
    if isinstance(write.source, BuiltImage):
        return "docker_image"
    type_ = write.source.entry.type
    return canonical_type(type_) if write.kind is BundleKind.ARTIFACT else type_


def _changes(row: Mapping[str, Any], digest: Digest) -> list[str]:
    """Why ``digest`` differs from the one ``row`` recorded, input by input when the two can be compared."""
    if row["digest"] == digest.value:
        return []
    if row["inputs"] is None:
        return ["the ledger doesn't know what its last version was made from"]
    if row["scheme"] != SCHEME:
        return ["the ledger's digest scheme changed"]
    before, after = row["inputs"], digest.inputs
    reasons = []
    if before["type"] != after["type"]:
        reasons.append(f"type changed: {before['type']} → {after['type']}")
    if before.get("config") != after["config"]:
        reasons.append("config changed")
    for label, names in zip(("files added", "files removed", "files changed"), _diff(before["files"], after["files"])):
        if names:
            shown = ", ".join(names[:_LISTED]) + (f" and {len(names) - _LISTED} more" if len(names) > _LISTED else "")
            reasons.append(f"{label}: {shown}")
    needs_before, needs_after = before["needs"], after["needs"]
    added, removed, changed = _diff(needs_before, needs_after)
    reasons.extend(f"{name} is written anew (v{needs_before[name]} → v{needs_after[name]})" for name in changed)
    reasons.extend(f"now needs {name}" for name in added)
    reasons.extend(f"no longer needs {name}" for name in removed)
    refs_before, refs_after = before["store_refs"], after["store_refs"]
    reasons.extend(f"{name} has a new version in the store (v{refs_before[name]} → v{refs_after[name]})"
                   for name in _diff(refs_before, refs_after)[2])
    return reasons or ["its inputs changed"]


def _diff(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[list[str], list[str], list[str]]:
    """The keys only ``after`` has, only ``before`` has, and both have with different values, each sorted."""
    return (sorted(after.keys() - before.keys()), sorted(before.keys() - after.keys()),
            sorted(key for key in after.keys() & before.keys() if after[key] != before[key]))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_scalar).encode()


def _scalar(value: Any) -> str:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not part of a bundle's config")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# A build copies each file's permission bits into the image, so they're part of what it's made from; a file with the
# usual rw-r--r-- is hashed as its content alone.
_USUAL_MODE = 0o644


def _mode(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    return "" if mode == _USUAL_MODE else f"+{mode:o}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
