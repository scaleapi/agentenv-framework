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

#: Where the counts are kept, one small file per context, under the agent's user's home.
TURNS_DIR = Path.home() / ".local" / "state" / "agentenv" / "turns"


class SnapshotState(BaseModel):
    """What the SDK keeps in a snapshot beside the agent's own objects: the context's turn count. A newer SDK may add
    fields, which an older one reading the snapshot leaves alone."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    turns: int = Field(ge=0)


class TurnIndex:
    """Each context's count of started tasks. A count that can't be written stays in memory, with a warning, so a
    read-only home costs only the count's survival across a restart."""

    def __init__(self, folder: Path | None = None) -> None:
        self._folder = folder if folder is not None else TURNS_DIR
        self._counts: dict[str, int] = {}
        self._warned = False

    def next(self, context_id: str) -> int:
        """The starting task's turn index, the number of tasks the context started before it, counting this one."""
        index = self.count(context_id)
        self.set(context_id, index + 1)
        return index

    def count(self, context_id: str) -> int:
        if context_id not in self._counts:
            try:
                self._counts[context_id] = json.loads(self._path(context_id).read_text())["turns"]
            except FileNotFoundError:
                self._counts[context_id] = 0
            except OSError as exc:
                self._warn(exc)
                self._counts[context_id] = 0
        return self._counts[context_id]

    def set(self, context_id: str, count: int) -> None:
        self._counts[context_id] = count
        path = self._path(context_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_text(json.dumps({"context_id": context_id, "turns": count}))
            os.replace(part, path)
        except OSError as exc:
            self._warn(exc)

    def _path(self, context_id: str) -> Path:
        return self._folder / f"{hashlib.sha256(context_id.encode()).hexdigest()}.json"

    def _warn(self, exc: OSError) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("turn counts are kept in memory only, so a restart starts them again: %s", exc)
