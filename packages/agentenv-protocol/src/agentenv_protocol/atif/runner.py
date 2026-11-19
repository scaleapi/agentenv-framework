"""Running an adapter: feed one A2A context's records through it, a task at a time. The runner writes each task's
user step and final response from what the SDK knows, and hands every handler the AtifTurn it writes through. The SDK
runs it live and the adapter kit replays it offline, so the same records, fed in the same order, build the same
document."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

from .adapter import Adapter, AdapterSpec
from .builder import AtifTurn, Document, Media, TurnContext, timestamp
from .profile import AtifError


@dataclass(frozen=True)
class Fed:
    """One record as it was fed: a native record, or a record of the aux file ``aux``."""

    record: Any
    aux: str | None = None


@dataclass(frozen=True, kw_only=True)
class RecordedTask:
    """A stored task: its context, its records in the order they were fed, and how it ended: its state, final response,
    the final response's diagnostic DataParts and its error."""

    context: TurnContext
    records: Sequence[Fed] = ()
    state: str = "completed"
    final: Sequence[Any] | None = None
    ended_at: str | datetime | None = None
    data: Sequence[Mapping[str, Any]] | None = None
    error: Mapping[str, Any] | None = None

    @classmethod
    def from_served(cls, turn: Mapping[str, Any], native: Sequence[Any],
                    aux: Sequence[Mapping[str, Any]] = ()) -> RecordedTask:
        """The task an atif/1 final read describes: its ``turn_context``, its native records (the native final read's
        events) and its ``aux`` log, each aux record placed after the native records fed before it."""
        records: list[Fed] = []
        fed = 0
        for entry in aux:
            while fed < entry["after"]:
                records.append(Fed(native[fed]))
                fed += 1
            records.append(Fed(entry["record"], aux=entry["name"]))
        records.extend(Fed(record) for record in native[fed:])
        context = TurnContext(**{key: turn.get(key) for key in _CONTEXT_FIELDS},
                              inbound=_content_from_json(turn.get("inbound") or ()))
        return cls(context=context, records=records, state=turn["state"],
                   final=_content_from_json(turn["final"]) if turn.get("final") is not None else None,
                   ended_at=turn.get("ended_at"), data=turn.get("data"), error=turn.get("error"))


_CONTEXT_FIELDS = ("index", "context_id", "task_id", "message_id", "agent_name", "agent_version", "agent_model",
                   "started_at")


def served_turn(context: TurnContext, *, state: str, final: Sequence[Any] | None = None,
                ended_at: str | datetime | None = None, data: Sequence[Mapping[str, Any]] | None = None,
                error: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A task's TurnContext and how it ended, as JSON: what an atif/1 final read serves as ``turn_context``, so an
    offline replay has everything but the records. Media in the content are named by path, their bytes stored apart."""
    turn: dict[str, Any] = {}
    for key in _CONTEXT_FIELDS:
        value = getattr(context, key)
        if key == "started_at":
            value = timestamp(value)
        if value is not None:
            turn[key] = value
    turn["inbound"] = _content_json(context.inbound)
    turn["state"] = state
    for key, value in (("final", None if final is None else _content_json(final)), ("ended_at", timestamp(ended_at)),
                       ("data", None if data is None else [dict(entry) for entry in data]),
                       ("error", None if error is None else dict(error))):
        if value is not None:
            turn[key] = value
    return json.loads(json.dumps(turn, allow_nan=False))


def _content_json(content: Sequence[Any]) -> list[Any]:
    return [{"media": {"path": item.path, "media_type": item.media_type, "info": dict(item.info)}}
            if isinstance(item, Media) else item for item in content]


def _content_from_json(content: Sequence[Any]) -> list[Any]:
    return [Media(item["media"]["path"], item["media"]["media_type"], item["media"]["info"])
            if isinstance(item, Mapping) and set(item) == {"media"} else item for item in content]


class AdapterRun:
    """One A2A context's conversion. ``page`` is the live view, the closed steps a reader may be served, built when
    it's read; ``document`` is everything so far. Every read is a fresh copy. ``media`` receives each media file once,
    as ``(path, bytes)``; without it, the files are kept in ``media_files``."""

    def __init__(self, adapter: type[Adapter], *, media: Callable[[str, bytes], None] | None = None,
                 trajectory_id: str | None = None) -> None:
        spec = vars(adapter).get("spec")
        if not isinstance(spec, AdapterSpec):
            raise AtifError(f"{adapter.__name__} isn't declared with @atif_adapter")
        self._cls = adapter
        self._spec = spec
        self._doc = Document(spec, media, trajectory_id=trajectory_id)
        self._adapter: Adapter | None = None
        self._turn: AtifTurn | None = None

    @property
    def media_files(self) -> dict[str, bytes]:
        return dict(self._doc.media)

    def media(self, data: bytes, media_type: str, *, width: int | None = None, height: int | None = None,
              duration_ms: int | None = None, pages: int | None = None, name: str | None = None) -> Media:
        """Store media for a TurnContext's inbound content or a final response, as a handler's ``turn.media``
        does."""
        return self._doc.store(data, media_type, {"width": width, "height": height, "duration_ms": duration_ms,
                                                  "pages": pages, "name": name})

    def document(self, *, through: int | None = None) -> dict[str, Any] | None:
        """The document so far; with ``through``, as it stood when this run's task ``through`` (from 0) ended."""
        return self._doc.document(through=through)

    def page(self) -> dict[str, Any] | None:
        return self._doc.page()

    def start_task(self, context: TurnContext) -> None:
        """Open a task and write its user step. The context's first task creates the adapter."""
        if self._turn is not None:
            raise AtifError(f"task {self._turn.context.task_id} is still open")
        turn = self._doc.start_task(context)
        self._turn = turn
        if self._adapter is None:
            self._adapter = self._cls(turn)

    def native(self, record: Any) -> None:
        """Feed the task's next native record."""
        turn = self._open()
        method = self._spec.handler(None, record) or "unmatched"
        getattr(self._adapter, method)(record, turn)

    def aux(self, name: str, record: Any) -> None:
        """Feed the task's next record of aux file ``name``."""
        turn = self._open()
        if name not in self._spec.aux:
            raise AtifError(f"{self._cls.__name__} declares no aux file {name!r}")
        method = self._spec.handler(name, record)
        if method is not None:
            getattr(self._adapter, method)(record, turn)

    def end_task(self, *, state: str = "completed", final: Sequence[Any] | None = None,
                 ended_at: str | datetime | None = None, data: Sequence[Mapping[str, Any]] | None = None,
                 error: Mapping[str, Any] | None = None) -> None:
        """Close the task: run the adapter's ``@finish``, then write the final response, once, and on the task's last
        step its terminal state, the final response's diagnostic DataParts (``data``: ``{data, metadata}`` each) and
        its error (``kind``, ``message``, ``code``) unless the adapter wrote one there. The task is closed even when
        ``@finish`` raises, and the error is raised after."""
        turn = self._open()
        try:
            if self._spec.finish is not None:
                getattr(self._adapter, self._spec.finish)(turn)
        finally:
            self._doc.end_task(state=state, final=final, ended_at=ended_at, data=data, error=error)
            self._turn = None

    def _open(self) -> AtifTurn:
        if self._turn is None:
            raise AtifError("no task is open: call start_task first")
        return self._turn


def replay(adapter: type[Adapter], tasks: Iterable[RecordedTask], *,
           media: Callable[[str, bytes], None] | None = None, trajectory_id: str | None = None) -> dict[str, Any] | None:
    """A context's stored tasks run through ``adapter`` again, as the agent ran them live: the whole document."""
    run = AdapterRun(adapter, media=media, trajectory_id=trajectory_id)
    for task in tasks:
        run.start_task(task.context)
        for fed in task.records:
            if fed.aux is None:
                run.native(fed.record)
            else:
                run.aux(fed.aux, fed.record)
        run.end_task(state=task.state, final=task.final, ended_at=task.ended_at, data=task.data, error=task.error)
    return run.document()
