"""The default data store of a simple site: named collections of JSON objects, each with an ``id``. Its data is a JSON
object of collections by name, beside which a string ``current_user`` names the user a browser starts signed in as;
the data plane loads, exports and resets it."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from ..types import DataPart, FilePart, Part, TextPart

CURRENT_USER = "current_user"


class Collections:
    """Named lists of JSON objects; ``store["orders"]`` is the list itself, created empty on first use."""

    def __init__(self, data: dict | None = None) -> None:
        self._collections: dict[str, list[dict]] = {}
        self.current_user: str | None = None
        if data:
            self.load(data)

    def __getitem__(self, name: str) -> list[dict]:
        return self._collections.setdefault(name, [])

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

    def export(self) -> dict:
        data: dict = copy.deepcopy(self._collections)
        if self.current_user is not None:
            data[CURRENT_USER] = self.current_user
        return data


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
