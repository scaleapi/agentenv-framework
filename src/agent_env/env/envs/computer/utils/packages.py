"""Fetch at put, install from the store: what a computer env's layers ask for beyond their own files (apt packages,
Python requirements and downloads) is resolved once for each arch in a container of the base (``resolve_in_base``),
every file is stored in the object store by its sha256, and a lock lists them (``Lock``). The lock is stored by a
digest of what it was resolved from (each layer's packages, sources, requirements file and downloads, the base, the
arch, the request the resolver is given, the resolver itself and the lock's own shape), so any env with the same inputs
reuses it. A build stages only the lock's files and installs them with the network closed (``composer``)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from agent_env.config import get_config
from agent_env.env.envs.computer.utils.builder import (
    CONTEXT, RESOLVED, RESOLVER, Builder, Hashed, ResolveError, sha256_of)
from agent_env.env.envs.computer.utils.install import Install
from agent_env.store.base import ObjectAlreadyExistsError

#: Where locks and their files are kept, under the configured key prefix, in the store the env's id routes to.
PREFIX = "agentenv-packages"
#: Where a build's context holds a lock's .debs; each venv's wheels are at ``staged_wheels``.
STAGED_APT = "packages/apt"
_RESOLVE_TIMEOUT_S = 3600


class LockedFile(BaseModel):
    file: str
    sha256: str
    size: int


class LockedDeb(LockedFile):
    package: str
    version: str
    arch: str


class LockedDownload(LockedFile):
    url: str
    to: str


class Lock(BaseModel):
    inputs: str
    arch: str
    apt: list[LockedDeb]
    wheels: dict[str, list[LockedFile]]
    downloads: dict[str, list[LockedDownload]]

    def digest(self) -> str:
        """The part of a build key that says exactly what the lock installs."""
        return "sha256:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


@dataclass(frozen=True)
class Want:
    """A layer's part of a lock: its folder in the build context, its install, and when it has a venv, the requirements
    the venv adds to its own (the protocol)."""

    folder: str
    install: Install
    venv: bool
    adds: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return _slug(self.folder)


def inputs_digest(wants: list[Want], context: Path, base: str, arch: str) -> str | None:
    """The digest of everything a lock of ``wants`` for ``arch`` is resolved from, their folders staged in ``context``
    and ``base`` the base's identity; None when they ask for nothing."""
    if not any(want.install.fetches() or want.venv for want in wants):
        return None
    layers = []
    for want in wants:
        install, folder = want.install, context / want.folder
        layers.append({
            "folder": want.folder, "apt": install.packages.apt if install.packages else [],
            "sources": [{**source.model_dump(), "key": sha256_of(folder / source.key)}
                        for source in install.sources or []],
            "python": sha256_of(folder / install.python) if install.python is not None else None,
            "venv": want.venv, "adds": list(want.adds),
            "downloads": [download.model_dump() for download in _downloads(install, arch)]})
    protocol = context / "protocol"
    return "sha256:" + hashlib.sha256(json.dumps({
        "layers": layers, "base": base, "arch": arch, "request": _request(wants, arch), "resolver": sha256_of(RESOLVER),
        "lock": _lock_shape(), "protocol": _folder_digest(protocol) if protocol.is_dir() else None,
    }, sort_keys=True).encode()).hexdigest()


def stored(inputs: str, store_id: str) -> Lock | None:
    """The lock resolved from ``inputs``, when one is stored."""
    store, prefix = _store(store_id)
    key = _lock_key(prefix, inputs)
    return Lock.model_validate_json(store.read(key)) if store.exists(key) else None


def resolve(inputs: str, wants: list[Want], builder: Builder, source: str, arch: str,
            store_id: str) -> tuple[Lock, dict[str, str]]:
    """Resolve ``wants`` for ``arch`` with ``builder``, in a container of ``source``, the base's image, with their
    folders and the protocol's source from its build context; fetch their downloads; store every file and the lock. The
    lock, and where on the builder each file this resolve fetched is, by its sha256, for a build that follows to stage
    it from there rather than the store."""
    (store, prefix), fetched = _store(store_id), {}
    result = builder.resolve(_request(wants, arch), source, arch, _RESOLVE_TIMEOUT_S)
    written = builder.files(f"{RESOLVED}/out")

    def put(folder: str, path: str) -> dict:
        return _put(builder, store, prefix, fetched, written, folder, path)

    apt = [LockedDeb(**{**deb, **put("out", f"apt/{deb['file']}")}) for deb in result["apt"]]
    wheels = {want.folder: [LockedFile(**put("out", f"wheels/{want.name}/{wheel}"))
                            for wheel in result["wheels"][want.name]] for want in wants if want.venv}
    for want in wants:
        for download in _downloads(want.install, arch):
            builder.download(download.url, download.sha256, f"{RESOLVED}/downloads/{download.sha256}")
    written = builder.files(f"{RESOLVED}/downloads") if any(_downloads(want.install, arch) for want in wants) else {}
    downloads = {want.folder: [LockedDownload(url=download.url, to=download.to, **put("downloads", download.sha256))
                               for download in _downloads(want.install, arch)]
                 for want in wants if _downloads(want.install, arch)}
    lock = Lock(inputs=inputs, arch=arch, apt=apt, wheels=wheels, downloads=downloads)
    key = _lock_key(prefix, inputs)
    try:
        store.put(key, lock.model_dump_json().encode(), content_type="application/json")
    except ObjectAlreadyExistsError:
        return Lock.model_validate_json(store.read(key)), fetched
    return lock, fetched


def stage(lock: Lock, builder: Builder, store_id: str, fetched: dict[str, str]) -> None:
    """Write the lock's files into ``builder``'s build context: .debs under ``STAGED_APT``, each venv's wheels under
    ``staged_wheels``, and each layer's downloads at their places in its folder, a script or binary executable; each
    linked from ``fetched``, the files a resolve just fetched by sha256, or else fetched from the store and refused
    unless its sha256 is the one the lock names."""
    store, prefix = _store(store_id)
    places = {f"{CONTEXT}/{STAGED_APT}/{deb.file}": deb.sha256 for deb in lock.apt}
    places |= {f"{CONTEXT}/{staged_wheels(folder)}/{wheel.file}": wheel.sha256
               for folder, wheels in lock.wheels.items() for wheel in wheels}
    downloads = {f"{CONTEXT}/{folder}/{download.to}": download.sha256
                 for folder, downloads in lock.downloads.items() for download in downloads}
    places |= downloads
    builder.place({target: fetched[sha256] for target, sha256 in places.items() if sha256 in fetched},
                  {target: (store.object_url(f"{prefix}/files/{sha256}"), sha256)
                   for target, sha256 in places.items() if sha256 not in fetched})
    builder.mark_runnable(list(downloads))


def staged_wheels(folder: str) -> str:
    """Where a build's context holds the wheels of the venv of the layer in ``folder``."""
    return f"packages/wheels/{_slug(folder)}"


def _request(wants: list[Want], arch: str) -> dict:
    """What the resolver in the base resolves, by its paths there: the build context at ``/ctx``."""
    apt = list(dict.fromkeys(name for want in wants for name in (want.install.packages.apt if want.install.packages
                                                                 else [])))
    sources = [{**source.model_dump(), "key": f"/ctx/{want.folder}/{source.key}"}
               for want in wants for source in want.install.sources or []]
    python = [{"name": want.name, "folder": f"/ctx/{want.folder}", "requirements": want.install.python,
               "add": list(want.adds)} for want in wants if want.venv]
    return {"apt": apt, "sources": sources, "python": python}


def _downloads(install: Install, arch: str) -> list:
    return [download for download in install.downloads or [] if download.arch in (None, arch)]


def _put(builder: Builder, store, prefix: str, fetched: dict[str, str], written: dict[str, Hashed], folder: str,
         path: str) -> dict:
    """Store the file at ``path`` in the resolve's ``folder`` by its sha256 unless it's there, and add it to
    ``fetched``; its name, sha256 and size. Refused unless it's one of the plain files ``written`` there."""
    if (hashed := written.get(path)) is None:
        raise ResolveError(f"the resolver's {path!r} isn't a plain file it wrote")
    key, at = f"{prefix}/files/{hashed.sha256}", f"{RESOLVED}/{folder}/{path}"
    if not store.exists(key):
        try:
            builder.store(at, store, key)
        except ObjectAlreadyExistsError:
            pass
    fetched[hashed.sha256] = at
    return {"file": PurePosixPath(path).name, "sha256": hashed.sha256, "size": hashed.size}


def _store(store_id: str):
    config = get_config()
    return config.get_object_store_for(store_id), f"{config.get_artifact_key_prefix()}{PREFIX}"


def _folder_digest(folder: Path) -> str:
    hashed = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        hashed.update(path.relative_to(folder).as_posix().encode() + b"\0" + bytes.fromhex(sha256_of(path)))
    return hashed.hexdigest()


def _lock_shape() -> dict[str, dict[str, str]]:
    """The fields of a lock and their types, one of its inputs, so a lock stored in another shape is resolved again."""
    return {model.__name__: {name: str(field.annotation) for name, field in model.model_fields.items()}
            for model in (Lock, LockedFile, LockedDeb, LockedDownload)}


def _lock_key(prefix: str, inputs: str) -> str:
    return f"{prefix}/locks/{inputs.removeprefix('sha256:')}.json"


def _slug(folder: str) -> str:
    return folder.replace("/", "-")
