"""A computer env's ``composed`` metadata: each build's result by form and arch, recorded as it finishes, with the
digest of the code that composed it, so a build an agent-env that composes differently made is composed again."""

from __future__ import annotations

import copy
import functools
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import agentenv_protocol.computer

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.env.env import Env
from agent_env.store.base import ConcurrentModificationError
from agent_env.store.ids import derive_id

if TYPE_CHECKING:
    from agent_env.env.envs.computer.env import ComputerEnv

# A pair's statuses once its image is in the store.
BUILT = ("unverified", "ready")
#: The image store repository composed computers are pushed to, tagged by their keys.
REPOSITORY = "agentenv-computers"
_WRITE_TRIES = 4
#: The code that decides what a composed computer holds: the composer, the computers' build files, the provider that
#: composes VM bundles, and the protocol's computer package each layer installs.
_PACKAGE = Path(__file__).resolve().parents[4]
COMPOSING = (Path(__file__).parent, _PACKAGE / "computers", _PACKAGE / "providers" / "sandbox_providers" / "lume",
             Path(agentenv_protocol.computer.__file__).parent)


class Record:
    """The env's ``composed`` metadata, by form and arch. Each result is merged into the stored table pair by pair, the
    table read again when another build wrote it first, so builds of one env at once keep each other's pairs, and a
    result never undoes one that got further (``_rank``)."""

    def __init__(self, env: ComputerEnv):
        self.env, self.lock = env, threading.Lock()

    def started(self, pairs: list[tuple[str, str]]) -> None:
        at = now()
        self._merge({pair: {"status": "building", "started_at": at, "composer": composition()} for pair in pairs})

    def finished(self, form: str, arch: str, result: dict) -> None:
        self._merge({(form, arch): {**result, "composer": composition()}})

    def built(self, form: str = "container") -> dict[str, str]:
        """Each arch's image of ``form``, of the pairs built or reused by this agent-env's composer."""
        return {arch: result["image"] for arch, result in self.env.metadata.get("composed", {}).get(form, {}).items()
                if result.get("status") in BUILT and not stale(result)}

    def computer(self, ref: dict | None) -> None:
        """Record the env's computer, with the images it was written over, the disks' and the VM bundles' apart, and the
        shared memory its apps need."""
        if ref is not None:
            with self.lock:
                disks, bundles, shm_mb = self.built("disk"), self.built("vm_bundle"), self.env.shm_mb()
                self.env.merge_metadata({"computer": {**ref, "images": self.built(), **({"disks": disks} if disks else {}),
                                                      **({"vm_bundles": bundles} if bundles else {}),
                                                      **({"shm_mb": shm_mb} if shm_mb else {})}})

    def _merge(self, results: dict[tuple[str, str], dict]) -> None:
        with self.lock:
            for attempt in range(_WRITE_TRIES):
                composed = copy.deepcopy(self.env.metadata.get("composed", {}))
                for (form, arch), result in results.items():
                    stored = composed.setdefault(form, {}).get(arch, {})
                    if _rank(result) >= _rank(stored):
                        composed[form][arch] = {"started_at": stored.get("started_at"), **result}
                try:
                    self.env.update_metadata({**self.env.metadata, "composed": composed})
                    return
                except ConcurrentModificationError:
                    if attempt == _WRITE_TRIES - 1:
                        raise
                    self.env.metadata = Env.get(self.env.stored_id, self.env.version).metadata


def failed(env: ComputerEnv, form: str, arch: str, stage: str, key: str | None, reason: str,
           log: str | None = None) -> dict:
    result = {"status": "failed", "stage": stage, "reason": reason, "failed_at": now()}
    if key is not None:
        result["key"] = key
    if log is not None:
        stored = FileArtifact.put_bytes(derive_id(env.stored_id, f"build_log_{form}_{arch}"), description=(
            f"the {form}/{arch} build of {env.stored_id} v{env.version}, which failed"), filename="build.log",
            content=log.encode(), content_type="text/plain")
        result["log"] = {"id": stored.id, "version": stored.version}
        result["reason"] = f"{reason}: {log.strip().splitlines()[-1] if log.strip() else 'no output'}"
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stale(result: dict | None) -> bool:
    """Whether ``result`` was recorded by an agent-env whose composing code differs from this one's, or by one from
    before results named it: what it holds may not be what this agent-env composes."""
    return result is not None and result.get("composer") != composition()


@functools.cache
def composition() -> str:
    """The digest of ``COMPOSING``'s files but their docs, by their paths under each folder, bytes and whether each is
    executable, as a build's key takes its context."""
    digest = hashlib.sha256()
    for folder in COMPOSING:
        for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix != ".md"
                           and "__pycache__" not in p.parts):
            digest.update(f"{folder.name}/{path.relative_to(folder).as_posix()}\0".encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest() + (b"x" if path.stat().st_mode & 0o100 else b"-"))
    return "sha256:" + digest.hexdigest()


def mirror_tag(key: str) -> str:
    """The tag of the image of ``key`` in a computer image mirror: the key, under the start of ``composition()``, so a
    change to the code that composes an image builds it again rather than reuse one the old code built."""
    return f"{composition().removeprefix('sha256:')[:16]}-{key.removeprefix('sha256:')}"


def _rank(result: dict) -> int:
    """How far a pair got: ready, then built and either unprobed or failed at its probe, then anything short of an
    image, and below that one another composer recorded. A result takes a stored one's place only when it got at least
    as far."""
    if result and stale(result):
        return -1
    if result.get("status") == "ready":
        return 2
    return 1 if result.get("status") == "unverified" or result.get("stage") == "probe" else 0
