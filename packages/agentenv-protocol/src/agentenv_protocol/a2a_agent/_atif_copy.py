"""The SDK's atif/1 copy of an A2A context: one adapter run per context, fed every native and aux record as the agent
writes it, and read as a task's document, the context's live page or a cursor page. An adapter that fails stops that
context's copy, never the agent's turn."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

from ..atif import FORMAT, PROFILE_KEY, Adapter, AdapterRun, Media, TurnContext, served_turn
from ..errors import AgentEnvError
from .tasks.v1 import DataPart, FilePart, TaskError, TextPart, _thaw

logger = logging.getLogger(__name__)

#: The DataPart key ``add_structured_output`` writes: its value is the task's answer, so it is written as text.
STRUCTURED_OUTPUT = "structured_output"
#: How a context's copy answers once its adapter failed: the request was fine, the copy can't be served.
UNAVAILABLE_STATUS = 409


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnStart:
    """What the SDK knows of a task as it starts, beside its ids: the inbound message and the agent."""

    message_id: str | None
    parts: Sequence[Any]
    agent_name: str | None
    agent_version: str | None
    agent_model: str | None


@dataclass(slots=True)
class _Task:
    place: int
    context: TurnContext
    natives: int = 0
    aux: list[dict[str, Any]] = field(default_factory=list)
    turn: dict[str, Any] | None = None
    ended: bool = False


@dataclass(frozen=True, slots=True)
class AtifPage:
    """A cursor page of a copy: the closed steps after ``after``, ``next`` the last one's ``step_id``, and the
    sub-agents of their tasks."""

    header: dict[str, Any] | None
    subagents: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    next: int
    has_more: bool


class AtifCopy:
    """One A2A context's atif/1 document. Safe to feed from the agent's thread while the loop reads it."""

    def __init__(self, adapter: type[Adapter], context_id: str) -> None:
        self._run = AdapterRun(adapter, trajectory_id=uuid.uuid4().hex)
        self._adapter = adapter.__name__
        self._context_id = context_id
        self._lock = threading.Lock()
        self._tasks: dict[str, _Task] = {}
        self._failed: str | None = None

    def start(self, task_id: str, index: int, turn: TurnStart) -> None:
        with self._lock:
            if self._failed is not None:
                return
            try:
                context = TurnContext(
                    index=index, context_id=self._context_id, task_id=task_id, message_id=turn.message_id,
                    inbound=tuple(_content(turn.parts, self._run)[0]), agent_name=turn.agent_name,
                    agent_version=turn.agent_version, agent_model=turn.agent_model, started_at=_now())
                self._run.start_task(context)
            except Exception as exc:
                self._fail(exc)
                return
            self._tasks[task_id] = _Task(place=len(self._tasks), context=context)

    def native(self, task_id: str, record: Any) -> None:
        with self._lock:
            task = self._open(task_id)
            if task is not None:
                self._feed(task, record)

    def aux(self, task_id: str, name: str, record: Any) -> None:
        with self._lock:
            task = self._open(task_id)
            if task is None:
                return
            # A copy: an adapter may change the record it's handed, and the log is replay's input.
            task.aux.append({"name": name, "record": json.loads(json.dumps(record)), "after": task.natives})
            try:
                self._run.aux(name, record)
            except Exception as exc:
                self._fail(exc)

    def end(self, task_id: str, *, state: str, final: Sequence[Any] = (), error: TaskError | None = None,
            native: Any = None) -> None:
        """Close the task: a native trajectory it returned instead of appending runs through the adapter first."""
        with self._lock:
            task = self._open(task_id)
            if task is None:
                return
            try:
                for record in (native if isinstance(native, list) else [native]) if native is not None else ():
                    self._feed(task, record)
                    if self._failed is not None:
                        return
                content, data = _content(final, self._run, answer=True)
                failure = None if error is None else {"kind": error.error_type, "message": error.message,
                                                      "code": error.code}
                ended = _now()
                ending = {"state": state, "final": content or None, "ended_at": ended, "data": data or None,
                          "error": failure}
                self._run.end_task(**ending)
                task.turn = served_turn(task.context, **ending)
            except Exception as exc:
                self._fail(exc)
            finally:
                task.ended = True

    def task_read(self, task_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None:
        """An ended task's document as it stood when the task ended, its aux log and its served turn context."""
        with self._lock:
            self._check()
            task = self._tasks.get(task_id)
            if task is None or task.turn is None:
                return None
            document = self._run.document(through=task.place)
            assert document is not None
            return document, json.loads(json.dumps(task.aux)), dict(task.turn)

    def context_read(self) -> dict[str, Any] | None:
        """The context's live page: every closed step a reader may be served, or None before the first."""
        with self._lock:
            self._check()
            return self._run.page()

    def page(self, after: int, max_steps: int, max_bytes: int, *, task_id: str | None = None) -> AtifPage:
        """The closed steps after step ``after``, of the whole context or of task ``task_id``, within both budgets; the
        first is always returned, so one larger than ``max_bytes`` can't stall a reader. With them, the sub-agents of
        every task they belong to, as far as each is served: a sub-agent can gain steps while its task runs, and the
        page that serves a task's last step, which closes only as the task ends, carries them whole."""
        with self._lock:
            self._check()
            document = self._run.page()
            places = {place: task_id for task_id, place in
                      ((task_id, task.place) for task_id, task in self._tasks.items())}
            sub_tasks = {agent_id: places.get(place) for agent_id, place in self._run.subagent_tasks().items()}
        steps = document["steps"] if document is not None else []
        if after > len(steps):
            raise ValueError(f"after={after} is past the document's {len(steps)} closed steps")
        owners = _owners(steps)
        scope = [(step, owner) for step, owner in zip(steps, owners) if task_id is None or owner == task_id]
        pending = [(step, owner) for step, owner in scope if step["step_id"] > after]
        served: list[dict[str, Any]] = []
        tasks: set[str | None] = set()
        used = 0
        for step, owner in pending[:max_steps]:
            size = len(json.dumps(step))
            if served and used + size > max_bytes:
                break
            served.append(step)
            tasks.add(owner)
            used += size
        header = None
        subagents: list[dict[str, Any]] = []
        if document is not None:
            header = {key: value for key, value in document.items() if key not in ("steps", "subagent_trajectories")}
            subagents = [sub for sub in document.get("subagent_trajectories", [])
                         if sub_tasks.get(sub["trajectory_id"]) in tasks]
        return AtifPage(header=header, subagents=subagents, steps=served,
                        next=served[-1]["step_id"] if served else after, has_more=len(served) < len(pending))

    def _open(self, task_id: str) -> _Task | None:
        if self._failed is not None:
            return None
        task = self._tasks.get(task_id)
        return task if task is not None and not task.ended else None

    def _feed(self, task: _Task, record: Any) -> None:
        try:
            self._run.native(record)
        except Exception as exc:
            self._fail(exc)
        task.natives += 1

    def _fail(self, exc: Exception) -> None:
        self._failed = f"{type(exc).__name__}: {exc}"
        logger.warning("the atif/1 copy of context %s stopped at an error in %s: %s", self._context_id, self._adapter,
                       self._failed)

    def _check(self) -> None:
        if self._failed is not None:
            raise AgentEnvError(
                "atif_unavailable",
                f"the {FORMAT} copy of context {self._context_id!r} stopped at an error in its adapter "
                f"{self._adapter}: {self._failed}; its native trajectory is unaffected",
                status=UNAVAILABLE_STATUS,
            )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _content(parts: Iterable[Any], run: AdapterRun, *, answer: bool = False) -> tuple[list[Any], list[dict[str, Any]]]:
    """A2A parts as ATIF content: text as text, a DataPart as its compact JSON, a file's bytes as media and a file by
    URI as a note naming it. In a final response (``answer``), a DataPart other than a structured output is the
    harness's diagnostics, so it goes to the data list for the profile instead."""
    content: list[Any] = []
    data: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append(part.text)
        elif isinstance(part, DataPart):
            value = _thaw(part.data)
            if answer and set(value) == {STRUCTURED_OUTPUT}:
                content.append(_compact(value[STRUCTURED_OUTPUT]))
            elif answer:
                entry: dict[str, Any] = {"data": value}
                if part.metadata:
                    entry["metadata"] = _thaw(part.metadata)
                data.append(entry)
            else:
                content.append(_compact(value))
        elif isinstance(part, FilePart):
            content.append(_file(part, run))
    return content, data


def _file(part: FilePart, run: AdapterRun) -> Media | str:
    media_type = part.mime_type or "application/octet-stream"
    if part.bytes is not None:
        try:
            data = base64.b64decode(part.bytes, validate=True)
        except (binascii.Error, ValueError):
            return f"[file: {part.name or 'unnamed'}, {media_type}, bytes that aren't base64]"
        return run.media(data, media_type, name=part.name)
    assert part.uri is not None
    where = urlsplit(part.uri)
    # The query and fragment are dropped: a signed URL's signature is a grant.
    return f"[file: {part.name or 'unnamed'}, {media_type}, {urlunsplit((*where[:3], '', ''))}]"


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _owners(steps: Sequence[dict[str, Any]]) -> list[str | None]:
    """The task each step belongs to: the one whose user step opened the run of steps it's in."""
    owners: list[str | None] = []
    owner = None
    for step in steps:
        if step.get("source") == "user":
            owner = ((step.get("extra") or {}).get(PROFILE_KEY) or {}).get("task_id", owner)
        owners.append(owner)
    return owners
