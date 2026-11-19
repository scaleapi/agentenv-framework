"""Each A2A context's turn count, kept on disk by the SDK, so a restarted server, or a context evicted from memory, goes
on counting where it was, and a snapshot carries it to the context it is restored into. Only an agent that serves
atif/1 keeps one: the count is its ATIF copy's turn index."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: The snapshot extension's card param naming the fields of the state an SDK keeps in each snapshot: a caller grants
#: the ``state`` object only to an agent whose card has it, since an SDK from before it refuses the object.
SNAPSHOT_STATE_PARAM = "state"


class SnapshotState(BaseModel):
    """What the SDK keeps in a snapshot beside the agent's own objects: the context's turn count. A newer SDK may add
    fields, which an older one reading the snapshot leaves alone."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    turns: int = Field(ge=0)


def turns_folder() -> Path:
    """Where the counts are kept, one small file per context: ``agentenv/turns`` in the user's state folder,
    ``$XDG_STATE_HOME`` when it's an absolute path, as the XDG spec has it, else ``~/.local/state``."""
    state = os.environ.get("XDG_STATE_HOME", "")
    return (Path(state) if os.path.isabs(state) else Path.home() / ".local" / "state") / "agentenv" / "turns"


class TurnIndex:
    """Each context's count of started tasks, read from its file each time, so two servers on one home agree and memory
    holds nothing per context. A count that can't be written stays in memory, with a warning, so a read-only home costs
    only the count's survival across a restart."""

    def __init__(self, folder: Path | None = None) -> None:
        self._folder = folder if folder is not None else turns_folder()
        self._unwritten: dict[str, int] = {}
        self._warned: set[str] = set()

    def next(self, context_id: str) -> int:
        """The starting task's turn index, the number of tasks the context started before it, counting this one."""
        index = self.count(context_id)
        self.set(context_id, index + 1)
        return index

    def count(self, context_id: str) -> int:
        if context_id in self._unwritten:
            return self._unwritten[context_id]
        try:
            return SnapshotState.model_validate_json(self._path(context_id).read_bytes()).turns
        except FileNotFoundError:
            return 0
        except (OSError, ValueError) as exc:
            self._warn("a turn count can't be read, so its context counts from 0 again: %s", exc)
            return 0

    def set(self, context_id: str, count: int) -> None:
        path = self._path(context_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_text(json.dumps({"context_id": context_id, "turns": count}))
            os.replace(part, path)
        except OSError as exc:
            self._warn("turn counts are kept in memory only, so a restart starts them again: %s", exc)
            self._unwritten[context_id] = count
        else:
            self._unwritten.pop(context_id, None)

    def _path(self, context_id: str) -> Path:
        return self._folder / f"{hashlib.sha256(context_id.encode()).hexdigest()}.json"

    def _warn(self, message: str, exc: Exception) -> None:
        if message not in self._warned:
            self._warned.add(message)
            logger.warning(message, exc)
