"""The Codex CLI's rollouts as its atif/1 adapter reads them: each record it reads as the ``rollout`` aux record
(``rollout_aux``), and the tail that finds them as a turn writes them (``RolloutTail``), its sub-agents' included."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROLLOUT_POLL_S = 0.05
# How often the follower looks for a sub-agent's new rollout, in polls.
_SUBAGENT_SCAN_POLLS = 10
_ROLLOUT_EVENTS = frozenset({"task_started", "task_complete"})
_ROLLOUT_ITEMS = frozenset({"reasoning", "message", "function_call", "custom_tool_call", "tool_search_call",
                            "web_search_call", "function_call_output", "custom_tool_call_output",
                            "tool_search_output"})


def rollout_aux(record: dict, thread_id: str, parent: str | None = None) -> dict | None:
    """A rollout record as the atif/1 adapter's ``rollout`` aux record, naming its thread and, for a sub-agent's, the
    thread that spawned it: what the adapter reads of it, or None for a record it doesn't read (the instructions and
    context the CLI sends, its own bookkeeping)."""
    kind, payload = record.get("type"), record.get("payload")
    if not isinstance(payload, dict):
        return None
    if kind == "session_meta":
        source = payload.get("source")
        spawn = source.get("subagent", {}).get("thread_spawn", {}) if isinstance(source, dict) else {}
        kept = {"id": payload.get("id"), "parent_thread_id": payload.get("parent_thread_id"),
                "agent_path": spawn.get("agent_path") if isinstance(spawn, dict) else None,
                "agent_nickname": payload.get("agent_nickname"), "cli_version": payload.get("cli_version")}
    elif kind == "turn_context":
        kept = {"model": payload.get("model"), "effort": payload.get("effort")}
    elif kind == "event_msg" and payload.get("type") in _ROLLOUT_EVENTS:
        kept = {key: payload.get(key) for key in ("type", "error")}
    elif kind == "token_usage_record":
        kept = {key: payload.get(key) for key in ("response_id", "usage")}
    elif kind == "response_item" and payload.get("type") in _ROLLOUT_ITEMS:
        if payload.get("type") == "message" and payload.get("role") != "assistant":
            return None
        kept = {key: value for key, value in payload.items() if key != "internal_chat_message_metadata_passthrough"}
    else:
        return None
    fed = {"type": kind, "timestamp": record.get("timestamp"), "thread": thread_id,
           "payload": {key: value for key, value in kept.items() if value is not None}}
    if parent is not None:
        fed["parent"] = parent
    return fed


@dataclass
class _Followed:
    thread: str
    parent: str | None
    inode: int
    offset: int
    lines: int
    # A forked sub-agent's rollout begins with the turns of the thread it was forked from, which aren't its own.
    forked_from: str | None = None
    copied_turns: set[str] = field(default_factory=set)
    own_turns: set[str] = field(default_factory=set)


def _turn_of(record: dict) -> str | None:
    """The turn a rollout record belongs to, as the CLI names it."""
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    passthrough = payload.get("internal_chat_message_metadata_passthrough")
    turn = payload.get("turn_id") or (passthrough.get("turn_id") if isinstance(passthrough, dict) else None)
    return turn if isinstance(turn, str) else None


def _thread_turns(find: Callable[[str], Path | None], thread: str) -> frozenset[str]:
    """Every turn a thread's rollout holds, the ones it copied from a thread it was forked from included."""
    path = find(thread)
    if path is None:
        return frozenset()
    turns = set()
    try:
        with path.open("rb") as file:
            for raw in file:
                if b'"task_started"' not in raw:
                    continue
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(record, dict) and (turn := _turn_of(record)) is not None:
                    turns.add(turn)
    except OSError:
        pass
    return frozenset(turns)


class RolloutTail:
    """One turn's new rollout records, as ``rollout_aux`` gives them: its thread's, and each sub-agent's it spawns at
    any depth, as the CLI appends them, from the rollouts under ``sessions``, ``find`` naming a thread's. Each file is
    read from where it ended when the turn began, so an earlier turn's records aren't fed again; a file replaced since
    is read on from its line count. ``bind`` names the thread once the CLI has started a new one."""

    def __init__(self, thread_id: str | None, sessions: Path, find: Callable[[str], Path | None]) -> None:
        self._lock = threading.Lock()
        self._sessions, self._find = sessions, find
        self._threads: set[str] = set()
        self._unfound: set[str] = set()
        self._files: dict[Path, _Followed] = {}
        self._metas: dict[Path, dict] = {}
        self._polls = 0
        if thread_id:
            self.bind(thread_id, from_end=True)
            self._scan(from_end=True)

    def bind(self, thread_id: str, *, from_end: bool = False) -> None:
        with self._lock:
            if thread_id in self._threads:
                return
            self._threads.add(thread_id)
            path = self._find(thread_id)
            if path is None:
                self._unfound.add(thread_id)
            else:
                self._start(path, thread_id, None, from_end)

    def read(self, *, final: bool = False) -> list[dict]:
        """The records the files gained since the last read, in the order they were written."""
        with self._lock:
            for thread in list(self._unfound):
                if (path := self._find(thread)) is not None:
                    self._unfound.discard(thread)
                    self._start(path, thread, None, from_end=False)
            self._polls += 1
            if final or self._polls % _SUBAGENT_SCAN_POLLS == 0:
                self._scan(from_end=False)
            records = [record for path in list(self._files) for record in self._new_records(path)]
        return sorted(records, key=lambda record: str(record.get("timestamp") or ""))

    def _start(self, path: Path, thread: str, parent: str | None, from_end: bool, forked_from: str | None = None
               ) -> None:
        try:
            status, lines = path.stat(), _count_lines(path) if from_end else 0
        except OSError:
            return
        self._files[path] = _Followed(thread, parent, status.st_ino, status.st_size if from_end else 0, lines,
                                      forked_from, set(_thread_turns(self._find, forked_from)) if forked_from else set())

    def _scan(self, *, from_end: bool) -> None:
        """Follow each sub-agent rollout not yet followed whose parent is a thread followed, until none is new. Each
        file's first record is read once, as it may name a parent followed only later."""
        if self._sessions.exists():
            for path in self._sessions.rglob("rollout-*.jsonl"):
                if path not in self._metas and (meta := _session_meta(path)) is not None:
                    self._metas[path] = meta
        found = True
        while found:
            found = False
            for path, meta in self._metas.items():
                thread, parent = meta.get("id"), meta.get("parent_thread_id")
                if parent in self._threads and isinstance(thread, str) and thread not in self._threads:
                    self._threads.add(thread)
                    forked = meta.get("forked_from_id")
                    self._start(path, thread, parent, from_end, forked if isinstance(forked, str) else None)
                    found = True

    def _new_records(self, path: Path) -> list[dict]:
        followed = self._files[path]
        try:
            status = path.stat()
            with path.open("rb") as file:
                if status.st_ino != followed.inode or status.st_size < followed.offset:
                    for _ in range(followed.lines):
                        if not file.readline():
                            break
                    followed.inode, followed.offset = status.st_ino, file.tell()
                file.seek(followed.offset)
                data = file.read()
        except OSError:
            return []
        complete = data[:data.rfind(b"\n") + 1]
        followed.offset += len(complete)
        records = []
        for raw in complete.splitlines():
            followed.lines += 1
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict) or self._copied(record, followed):
                continue
            if (fed := rollout_aux(record, followed.thread, followed.parent)) is not None:
                records.append(fed)
        return records


    def _copied(self, record: dict, followed: _Followed) -> bool:
        """A record a forked sub-agent's rollout copied from the thread it was forked from: that thread's meta, or
        one of its turns', which that thread's rollout is read again for when a turn is new here."""
        payload = record.get("payload")
        if record.get("type") == "session_meta":
            return isinstance(payload, dict) and payload.get("id") != followed.thread
        turn = _turn_of(record)
        if followed.forked_from is None or turn is None or turn in followed.own_turns:
            return False
        if turn not in followed.copied_turns:
            followed.copied_turns |= _thread_turns(self._find, followed.forked_from)
            if turn not in followed.copied_turns:
                followed.own_turns.add(turn)
                return False
        return True


def _count_lines(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for _ in file)


def _session_meta(path: Path) -> dict | None:
    """A rollout's first record's payload when it is the session's meta, None while the CLI hasn't written it."""
    try:
        with path.open("rb") as file:
            first = file.readline()
    except OSError:
        return None
    if not first.endswith(b"\n"):
        return None
    try:
        record = json.loads(first)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    payload = record.get("payload") if isinstance(record, dict) and record.get("type") == "session_meta" else None
    return payload if isinstance(payload, dict) else {}
