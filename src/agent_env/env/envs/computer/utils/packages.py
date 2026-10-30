"""Fetch at put, install from the store: what a computer env's layers ask for beyond their own files (apt packages,
Python requirements and downloads) is resolved once for each arch in a container of the base (``resolve_in_base``),
every file is stored in the object store by its sha256, and a lock lists them. The lock is stored by a digest of what
it was resolved from (each layer's packages, sources, requirements file and downloads, the base, the arch and this
resolver), so any env with the same inputs reuses it. A build stages only the lock's files and installs them with the
network closed (``composer``)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from agent_env.config import get_config
from agent_env.env.envs.computer.utils.install import Install
from agent_env.store.base import ObjectAlreadyExistsError

#: Where locks and their files are kept, under the configured key prefix, in the store the env's id routes to.
PREFIX = "agentenv-packages"
RESOLVER = Path(__file__).with_name("resolve_in_base.py")
#: What a build stages the lock's .debs and wheels under, in its context.
STAGED = "packages"
_RESOLVE_TIMEOUT_S = 3600
_DOWNLOAD_TIMEOUT_S = 600
_CHUNK = 1 << 20
#: What a file starts with when it's only ever run: a script or a Linux binary, made executable where it's staged,
#: since the stores keep no file modes.
RUN_ONLY = (b"#!", b"\x7fELF")
EXECUTABLE = 0o755


class ResolveError(RuntimeError):
    """Resolving or fetching failed; ``log`` holds what the resolver printed."""

    def __init__(self, message: str, log: str = ""):
        super().__init__(message)
        self.log = log


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
        return self.folder.replace("/", "-")


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
            "sources": [{**source.model_dump(), "key": _sha256(folder / source.key)} for source in install.sources or []],
            "python": _sha256(folder / install.python) if install.python is not None else None,
            "venv": want.venv, "adds": list(want.adds),
            "downloads": [download.model_dump() for download in _downloads(install, arch)]})
    protocol = context / "protocol"
    return "sha256:" + hashlib.sha256(json.dumps({
        "layers": layers, "base": base, "arch": arch, "resolver": _resolver_digest(),
        "protocol": _folder_digest(protocol) if protocol.is_dir() else None,
    }, sort_keys=True).encode()).hexdigest()


def stored(inputs: str, store_id: str) -> dict | None:
    """The lock resolved from ``inputs``, when one is stored."""
    store, prefix = _store(store_id)
    key = f"{prefix}/locks/{inputs.removeprefix('sha256:')}.json"
    return json.loads(store.read(key)) if store.exists(key) else None


def resolve(inputs: str, wants: list[Want], context: Path, source: str, arch: str, store_id: str,
            work: Path) -> tuple[dict, dict[str, Path]]:
    """Resolve ``wants`` for ``arch`` in a container of ``source``, the base's image, with their folders and the
    protocol's source from ``context``, into ``work``; fetch their downloads; store every file and the lock. The lock,
    and where each file this resolve fetched is under ``work``, by its sha256, for a build that follows to stage it from
    there rather than the store."""
    (store, prefix), fetched = _store(store_id), {}
    result, out = _run_resolver(_request(wants, arch), context, source, arch, work)
    apt = [{**deb, **_put(store, prefix, fetched, _written(out, "apt", deb["file"]))} for deb in result["apt"]]
    wheels = {want.folder: [_put(store, prefix, fetched, _written(out, "wheels", want.name, wheel))
                            for wheel in result["wheels"][want.name]] for want in wants if want.venv}
    downloads = {want.folder: [{"url": download.url, "to": download.to,
                                **_put(store, prefix, fetched, _fetch(download.url, download.sha256, work))}
                               for download in _downloads(want.install, arch)]
                 for want in wants if _downloads(want.install, arch)}
    lock = {"inputs": inputs, "arch": arch, "apt": apt, "wheels": wheels, "downloads": downloads}
    key = f"{prefix}/locks/{inputs.removeprefix('sha256:')}.json"
    try:
        store.put(key, json.dumps(lock, sort_keys=True).encode(), content_type="application/json")
    except ObjectAlreadyExistsError:
        return json.loads(store.read(key)), fetched
    return lock, fetched


def digest(lock: dict | None) -> str | None:
    """The digest of ``lock``, the part of a build key that says exactly what it installs."""
    return None if lock is None else "sha256:" + hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()


def stage(lock: dict, context: Path, store_id: str, fetched: dict[str, Path]) -> None:
    """Write the lock's files into ``context`` for a build: .debs under ``packages/apt``, each layer's wheels under
    ``packages/wheels/<name>``, and its downloads at their places in its folder, a script or binary executable; each
    linked from ``fetched``, the files a resolve just fetched by sha256, or else downloaded from the store and refused
    unless its sha256 is the one the lock names."""
    store, prefix = _store(store_id)
    for deb in lock["apt"]:
        _get(store, prefix, fetched, deb["sha256"], context / STAGED / "apt" / deb["file"])
    for folder, wheels in lock["wheels"].items():
        for wheel in wheels:
            _get(store, prefix, fetched, wheel["sha256"],
                 context / STAGED / "wheels" / folder.replace("/", "-") / wheel["file"])
    for folder, downloads in lock["downloads"].items():
        for download in downloads:
            target = context / folder / download["to"]
            _get(store, prefix, fetched, download["sha256"], target)
            with target.open("rb") as file:
                if file.read(4).startswith(RUN_ONLY):
                    target.chmod(EXECUTABLE)


def _request(wants: list[Want], arch: str) -> dict:
    """What the resolver in the base resolves, by its paths there: the build context at ``/ctx``."""
    apt = list(dict.fromkeys(name for want in wants for name in (want.install.packages.apt if want.install.packages
                                                                 else [])))
    sources = [{**source.model_dump(), "key": f"/ctx/{want.folder}/{source.key}"}
               for want in wants for source in want.install.sources or []]
    python = [{"name": want.name, "folder": f"/ctx/{want.folder}", "requirements": want.install.python,
               "add": list(want.adds)} for want in wants if want.venv]
    return {"apt": apt, "sources": sources, "python": python, "owner": [os.getuid(), os.getgid()]}


def _run_resolver(request: dict, context: Path, source: str, arch: str, work: Path) -> tuple[dict, Path]:
    """Run the resolver in a container of ``source`` for ``arch``, as root, with the build context read-only at /ctx;
    what it wrote, and the folder it wrote it in."""
    out = work / "out"
    out.mkdir(parents=True)
    (out / "request.json").write_text(json.dumps(request))
    shutil.copy(RESOLVER, out / "resolve.py")
    try:
        ran = subprocess.run(["docker", "run", "--rm", "--platform", f"linux/{arch}", "--user", "root",
                              "--entrypoint", "python3", "-v", f"{context}:/ctx:ro", "-v", f"{out}:/out", source,
                              "/out/resolve.py", "/out/request.json", "/out"],
                             capture_output=True, text=True, timeout=_RESOLVE_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise ResolveError(f"resolving took over {_RESOLVE_TIMEOUT_S} s", str(e.stdout or "")) from None
    if ran.returncode != 0:
        raise ResolveError("resolving the packages failed", ran.stdout + ran.stderr)
    return json.loads((out / "result.json").read_text()), out


def _fetch(url: str, sha256: str, work: Path) -> Path:
    """Download ``url`` into ``work``, refused unless its sha256 is ``sha256``."""
    target, hashed = work / "downloads" / sha256, hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            response.raise_for_status()
            with target.open("wb") as file:
                for chunk in response.iter_bytes(_CHUNK):
                    hashed.update(chunk)
                    file.write(chunk)
    except httpx.HTTPError as e:
        raise ResolveError(f"downloading {url} failed: {e}") from None
    if hashed.hexdigest() != sha256:
        raise ResolveError(f"{url} has the sha256 {hashed.hexdigest()}, not the {sha256} its download names")
    return target


def _downloads(install: Install, arch: str) -> list:
    return [download for download in install.downloads or [] if download.arch in (None, arch)]


def _written(out: Path, *parts: str) -> Path:
    """The file the resolver says it wrote at ``parts`` under ``out``, refused unless it's a plain file there: a
    dependency's build code, run in the container, could leave a link to one of this machine's files instead."""
    path = out.joinpath(*parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(out.resolve()):
        raise ResolveError(f"the resolver's {'/'.join(parts)!r} isn't a plain file it wrote")
    return path


def _put(store, prefix: str, fetched: dict[str, Path], path: Path) -> dict:
    """Store the file at ``path`` by its sha256 unless it's there, and add it to ``fetched``; its name, sha256 and
    size."""
    sha256 = _sha256(path)
    key = f"{prefix}/files/{sha256}"
    if not store.exists(key):
        try:
            store.put_file(key, str(path))
        except ObjectAlreadyExistsError:
            pass
    fetched[sha256] = path
    return {"file": path.name, "sha256": sha256, "size": path.stat().st_size}


def _get(store, prefix: str, fetched: dict[str, Path], sha256: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if sha256 in fetched:
        target.unlink(missing_ok=True)
        os.link(fetched[sha256], target)
        return
    key = f"{prefix}/files/{sha256}"
    store.download_to_file(store.object_url(key), str(target))
    if (found := _sha256(target)) != sha256:
        raise RuntimeError(f"the store's {key} isn't the file the lock names: its sha256 is {found}")


def _store(store_id: str):
    config = get_config()
    return config.get_object_store_for(store_id), f"{config.get_artifact_key_prefix()}{PREFIX}"


def _sha256(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            hashed.update(chunk)
    return hashed.hexdigest()


def _folder_digest(folder: Path) -> str:
    hashed = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        hashed.update(path.relative_to(folder).as_posix().encode() + b"\0" + bytes.fromhex(_sha256(path)))
    return hashed.hexdigest()


def _resolver_digest() -> str:
    return hashlib.sha256(RESOLVER.read_bytes() + Path(__file__).read_bytes()).hexdigest()
