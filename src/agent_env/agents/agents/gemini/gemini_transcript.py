"""The Gemini CLI's session transcripts as its atif/1 adapter reads them: each model message it reads as the
``session`` aux record (``session_aux``), and the tail that finds them as a turn writes them (``SessionTail``), from a
``.json`` transcript the CLI rewrites whole or a ``.jsonl`` one it appends to."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

SESSION_POLL_S = 0.25


def session_aux(message: dict) -> dict | None:
    """A session transcript's model message as the atif/1 adapter's ``session`` aux record: what the adapter reads of
    it, or None for a message it doesn't read (the user's, the CLI's notices). Each call keeps its result, without how
    the CLI displays it."""
    if message.get("type") != "gemini":
        return None
    record = {key: message[key] for key in ("type", "id", "timestamp", "model", "content", "thoughts", "tokens")
              if message.get(key) is not None}
    calls = [{key: call[key] for key in ("id", "name", "args", "status", "timestamp", "result")
              if call.get(key) is not None} for call in message.get("toolCalls") or () if isinstance(call, dict)]
    if calls:
        record["toolCalls"] = calls
    return record


def transcript_messages(text: str, jsonl: bool) -> list[dict] | None:
    """A transcript's messages in order: a ``.json`` file's ``messages``, which the CLI rewrites whole, or a ``.jsonl``
    file's lines after its first, the session's metadata, a message written again replacing its earlier copy in its
    place. None for a ``.json`` file that doesn't parse whole, as one the CLI is writing; a ``.jsonl`` line not yet
    whole is left out."""
    if not jsonl:
        try:
            transcript = json.loads(text)
        except ValueError:
            return None
        messages = transcript.get("messages") if isinstance(transcript, dict) else None
        return [message for message in messages if isinstance(message, dict)] if isinstance(messages, list) else None
    by_id: dict[str, dict] = {}
    for line in text.splitlines(keepends=True)[1:]:
        try:
            message = json.loads(line) if line.endswith("\n") else None
        except ValueError:
            continue
        if isinstance(message, dict) and isinstance(message.get("id"), str):
            by_id[message["id"]] = message
    return list(by_id.values())


class SessionTail:
    """One turn's new model messages from the session's transcript, ``.json`` or ``.jsonl``, as ``session_aux`` gives
    them: each once a later message follows it, since the CLI adds a message's tool calls and their results after
    writing it, and the last at the turn's end. A resumed session's messages from before the turn aren't fed again,
    one restored from a snapshot's included, ``find`` naming a session's transcript. ``bind`` names the session once the
    CLI has started a new one."""

    def __init__(self, session_id: str | None, find: Callable[[str], Path | None]) -> None:
        self._lock = threading.Lock()
        self._find = find
        self._session: str | None = None
        self._path: Path | None = None
        self._stamp: tuple[int, int] | None = None
        self._messages: list[dict] = []
        self._fed: set[str] = set()
        self._resumed = False
        if session_id:
            self.bind(session_id, resumed=True)

    def bind(self, session_id: str, *, resumed: bool = False) -> None:
        with self._lock:
            if self._session is not None:
                return
            self._session, self._resumed = session_id, resumed
            if resumed:
                self._load()

    def read(self, *, final: bool = False) -> list[dict]:
        """The messages settled since the last read, in order; with ``final``, the last one too."""
        with self._lock:
            self._load()
            settled = self._messages if final else self._messages[:-1]
            fed = []
            for message in settled:
                if not isinstance(message.get("id"), str) or message["id"] in self._fed:
                    continue
                self._fed.add(message["id"])
                if (record := session_aux(message)) is not None:
                    fed.append(record)
            return fed

    def _load(self) -> None:
        """The transcript's messages, read again when the file changed: one the CLI is still writing is read at the
        next poll, and one it moved, as from ``.json`` to ``.jsonl``, is found and read in this one."""
        if self._session is None:
            return
        for _ in range(2):
            if self._path is None:
                self._path = self._find(self._session)
                if self._path is None:
                    return
            try:
                status = self._path.stat()
                stamp = (status.st_mtime_ns, status.st_size)
                if stamp == self._stamp:
                    return
                text = self._path.read_text()
                break
            except FileNotFoundError:
                self._path = self._stamp = None
            except OSError:
                return
        else:
            return
        messages = transcript_messages(text, jsonl=self._path.suffix == ".jsonl")
        if messages is not None:
            self._stamp = stamp
            self._messages = messages
            if self._resumed:
                # The transcript as the turn found it: what's in it already is earlier turns'.
                self._fed.update(message.get("id") for message in self._messages)
                self._resumed = False
