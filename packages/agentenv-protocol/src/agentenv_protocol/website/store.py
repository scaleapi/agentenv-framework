"""The default state store of a simple site: named collections of JSON objects, each with an ``id``. Its data is a JSON
object of collections by name, beside which a string ``current_user`` names the user a browser starts signed in as.
It remembers the state it was last loaded with, so a verifier can read the initial state, the current one and the
difference between them.

A store also keeps sessions, as CUA-Gym sites do, so browsers that each name a session id (``?sid=``) never see one
another's changes: ``session(sid)`` is a store of its own, started on first use as a copy of this store's initial
state, and the session ``default``, or none, is this store itself. ``reset`` drops every session; a session started
before a later ``load`` keeps the state it was started with."""
from __future__ import annotations

import base64
import contextvars
import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from ..types import DataPart, FilePart, Part, TextPart

CURRENT_USER = "current_user"
DEFAULT_SID = "default"
SID_MAX = 128
_NOT_SID = re.compile(r"[^A-Za-z0-9_-]")
active_sid: contextvars.ContextVar[str | None] = contextvars.ContextVar("agentenv_website_sid", default=None)


def session_id(value: str | None) -> str | None:
    """The session id ``value`` names, as CUA-Gym sites read ``sid``: its letters, digits, ``_`` and ``-``, at most
    ``SID_MAX`` of them; None, the default session, for one that is empty or ``default``."""
    sid = _NOT_SID.sub("", value or "")[:SID_MAX]
    return None if sid in ("", DEFAULT_SID) else sid


class Collections:
    """Named lists of JSON objects; ``store["orders"]`` is the list itself, created empty on first use."""

    def __init__(self, data: dict | None = None) -> None:
        self._collections: dict[str, list[dict]] = {}
        self.current_user: str | None = None
        self.initial: dict = {}
        self._sessions: dict[str, Collections] = {}
        if data:
            self.load(data)
            self.mark_initial()

    def __getitem__(self, name: str) -> list[dict]:
        return self._collections.setdefault(name, [])

    def session(self, sid: str | None) -> Collections:
        """The store of session ``sid``, started as a copy of this one's initial state on first use; this store
        itself for the default session."""
        sid = session_id(sid)
        if sid is None:
            return self
        if sid not in self._sessions:
            self._sessions[sid] = Collections(self.initial)
        return self._sessions[sid]

    def sessions(self) -> list[str]:
        """The ids of the sessions started since the store was last reset, the default one aside."""
        return list(self._sessions)

    def changed_sessions(self) -> list[str]:
        """The ids of the sessions, the default one aside, whose state differs from the one they started with."""
        return [sid for sid, store in self._sessions.items() if store.diff()]

    def names(self) -> list[str]:
        return list(self._collections)

    def get(self, name: str, id: Any) -> dict | None:
        return next((doc for doc in self[name] if doc.get("id") == id), None)

    def find(self, name: str, **match: Any) -> list[dict]:
        return [doc for doc in self[name] if all(doc.get(k) == v for k, v in match.items())]

    def insert(self, name: str, doc: dict) -> dict:
        """Add ``doc``, giving it the next integer ``id`` of its collection when it has none."""
        doc = dict(doc)
        if doc.get("id") is None:
            doc["id"] = 1 + max((d["id"] for d in self[name] if isinstance(d.get("id"), int)), default=0)
        self[name].append(doc)
        return doc

    def update(self, name: str, id: Any, **fields: Any) -> dict | None:
        doc = self.get(name, id)
        if doc is not None:
            doc.update(fields)
        return doc

    def delete(self, name: str, id: Any) -> bool:
        doc = self.get(name, id)
        if doc is not None:
            self[name].remove(doc)
        return doc is not None

    def load(self, data: dict) -> None:
        """Add ``data``'s collections to the store's, and take its ``current_user`` when it names one."""
        if not isinstance(data, dict):
            raise ValueError("a store's data is a JSON object of collections by name")
        for name, docs in data.items():
            if name == CURRENT_USER:
                self.current_user = docs
            elif isinstance(docs, list) and all(isinstance(d, dict) for d in docs):
                self[name].extend(copy.deepcopy(docs))
            else:
                raise ValueError(f"collection {name!r} is not a list of JSON objects")

    def load_parts(self, parts: Iterable[Part]) -> None:
        for part in parts:
            self.load(part_data(part))

    def reset(self) -> None:
        self._collections.clear()
        self.current_user = None
        self.initial = {}
        self._sessions.clear()

    def export(self) -> dict:
        data: dict = copy.deepcopy(self._collections)
        if self.current_user is not None:
            data[CURRENT_USER] = self.current_user
        return data

    def mark_initial(self) -> None:
        self.initial = self.export()

    def diff(self) -> dict:
        return state_diff(self.initial, self.export())

    def state(self) -> dict:
        return verifier_state(self.initial, self.export())


def state_diff(initial: dict, current: dict, key: Any = None) -> dict:
    """Each collection that changed from ``initial`` to ``current``, as its ``added``, ``removed`` and ``changed``
    objects, matched by ``key(name, doc)``, their ``id`` by default; a changed one as its ``id``, ``before`` and
    ``after``."""
    key = key or (lambda name, doc: _key(doc))
    changes: dict = {}
    for name in dict.fromkeys([*initial, *current]):
        if name == CURRENT_USER:
            continue
        before = {key(name, d): d for d in initial.get(name, [])}
        after = {key(name, d): d for d in current.get(name, [])}
        entry = {
            "added": [d for k, d in after.items() if k not in before],
            "removed": [d for k, d in before.items() if k not in after],
            "changed": [{"id": d.get("id"), "before": before[k], "after": d}
                        for k, d in after.items() if k in before and before[k] != d],
        }
        if any(entry.values()):
            changes[name] = entry
    return changes


def verifier_state(initial: dict, current: dict, key: Any = None) -> dict:
    """What ``/go`` answers: ``{initial_state, current_state, state_diff}``."""
    return {"initial_state": copy.deepcopy(initial), "current_state": current,
            "state_diff": state_diff(initial, current, key)}


def _key(doc: dict) -> str:
    return json.dumps(doc.get("id")) if doc.get("id") is not None else json.dumps(doc, sort_keys=True)


def part_data(part: Part) -> dict:
    """A part's JSON: a data part's data, a text part's text, or a file part's bytes or ``file://`` file."""
    if isinstance(part, DataPart):
        return part.data
    if isinstance(part, TextPart):
        return json.loads(part.text)
    if isinstance(part, FilePart):
        file = part.file
        if getattr(file, "bytes", None) is not None:
            return json.loads(base64.b64decode(file.bytes))
        uri = urlparse(file.uri)
        if uri.scheme == "file":
            return json.loads(Path(unquote(uri.path)).read_text(encoding="utf-8"))
        raise ValueError(f"the store reads file parts by their bytes or a file:// URI, not {file.uri!r}")
    raise ValueError(f"the store can't read a {type(part).__name__}")
