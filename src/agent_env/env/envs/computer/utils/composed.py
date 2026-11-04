"""A computer env's ``composed`` metadata: each build's result by form and arch, recorded as it finishes."""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

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


class Record:
    """The env's ``composed`` metadata, by form and arch. Each result is merged into the stored table pair by pair, the
    table read again when another build wrote it first, so builds of one env at once keep each other's pairs, and a
    result never undoes one that got further (``_rank``)."""

    def __init__(self, env: ComputerEnv):
        self.env, self.lock = env, threading.Lock()

    def started(self, pairs: list[tuple[str, str]]) -> None:
        at = now()
        self._merge({pair: {"status": "building", "started_at": at} for pair in pairs})

    def finished(self, form: str, arch: str, result: dict) -> None:
        self._merge({(form, arch): result})

    def built(self, form: str = "container") -> dict[str, str]:
        """Each arch's image of ``form``, of the pairs built or reused."""
        return {arch: result["image"] for arch, result in self.env.metadata.get("composed", {}).get(form, {}).items()
                if result.get("status") in BUILT}

    def computer(self, ref: dict | None) -> None:
        """Record the env's computer, with the images it was written over, the disks' and the VM bundles' apart."""
        if ref is not None:
            with self.lock:
                disks, bundles = self.built("disk"), self.built("vm_bundle")
                self.env.merge_metadata({"computer": {**ref, "images": self.built(), **({"disks": disks} if disks else {}),
                                                      **({"vm_bundles": bundles} if bundles else {})}})

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


def _rank(result: dict) -> int:
    """How far a pair got: ready, then built and either unprobed or failed at its probe, then anything short of an
    image. A result takes a stored one's place only when it got at least as far."""
    if result.get("status") == "ready":
        return 2
    return 1 if result.get("status") == "unverified" or result.get("stage") == "probe" else 0
