"""Notes: the reference site of ``urn:agentenv:website/v1``, a small notes app on WebsiteEnvironment. Each user keeps
notes they create, edit, mark done and delete; a browser with no session is signed in as the acting user, the seed's
``current_user``, and can sign in as another seeded user. The site keeps its users and notes itself, behind one lock
since plain ``def`` handlers run in worker threads, starting from ``data/seed.json``, and serves them through the data plane (``data/get``, ``data/reset``, ``data/add``), as any
AgentEnvEnvironment does; it stamps each note with the universe's time. ``admin`` callers, as the gateway's
``AgentEnv-Role`` names them, can list every user's notes.

Run it with ``python site.py`` after building ``web/`` into ``web/dist``; it serves on ``PORT`` (8000)."""
from __future__ import annotations

import base64
import copy
import json
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from pydantic import BaseModel

from agentenv_protocol import DataPart, FilePart, TextPart, add_data, environment_card, get_data, reset_data
from agentenv_protocol.website import (
    ApiContext,
    BadRequest,
    NotFound,
    Unauthorized,
    WebsiteEnvironment,
    api,
    ready_check,
    website,
)
from agentenv_protocol.website.clock import iso

NOTES = "notes"
USERS = "users"
SEED = Path(__file__).resolve().parent / "data" / "seed.json"


class NewNote(BaseModel):
    title: str
    body: str = ""


class NoteEdit(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    done: Optional[bool] = None


class SignIn(BaseModel):
    user: str


@environment_card(name="notes")
@website(domains=["notes.test"], static="web/dist")
class Notes(WebsiteEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.data: dict = json.loads(SEED.read_text(encoding="utf-8"))

    def acting_user(self) -> Optional[str]:
        with self._lock:
            return self.data.get("current_user")

    @ready_check
    def seeded(self) -> bool:
        """The site is usable once it knows who may sign in."""
        with self._lock:
            return bool(self.data.get(USERS))

    @reset_data
    def reset(self) -> None:
        with self._lock:
            self.data = {}

    @add_data
    def add(self, parts: list) -> None:
        """Each part's JSON, its collections' records added by id and its other fields set, as the seed is."""
        loaded = [_part_data(part) for part in parts]
        with self._lock:
            for data in loaded:
                for key, value in data.items():
                    if isinstance(value, list):
                        kept = {record["id"]: record for record in self.data.get(key, [])}
                        kept.update({record["id"]: copy.deepcopy(record) for record in value})
                        self.data[key] = list(kept.values())
                    else:
                        self.data[key] = copy.deepcopy(value)

    @get_data
    def export(self) -> list:
        with self._lock:
            return [DataPart(data=copy.deepcopy(self.data))]

    @api.get("/api/users", name="list_users")
    def list_users(self) -> list:
        """The users a browser can sign in as."""
        with self._lock:
            return copy.deepcopy(self.data.get(USERS, []))

    @api.post("/api/sign-in", name="sign_in")
    def sign_in(self, body: SignIn, ctx: ApiContext) -> dict:
        """Sign the browser in as one of the site's users."""
        with self._lock:
            known = any(user["id"] == body.user for user in self.data.get(USERS, []))
        if not known:
            raise NotFound(f"no user {body.user!r}")
        ctx.sign_in(body.user)
        return {"user": ctx.user}

    @api.get("/api/notes", name="list_notes")
    def list_notes(self, ctx: ApiContext, done: Optional[bool] = None) -> list:
        """The signed-in user's notes, newest first; with ``done``, only those done or not."""
        user = self._user(ctx)
        with self._lock:
            mine = [copy.deepcopy(n) for n in self.data.get(NOTES, [])
                    if n["owner"] == user and (done is None or n["done"] == done)]
        return sorted(mine, key=lambda note: note["created_at"], reverse=True)

    @api.post("/api/notes", name="create_note", status=201)
    def create_note(self, note: NewNote, ctx: ApiContext) -> dict:
        """Add a note for the signed-in user, not done, created and updated now."""
        owner, title, now = self._user(ctx), _title(note.title), iso(self.now())
        with self._lock:
            notes = self.data.setdefault(NOTES, [])
            created = {"id": max((n["id"] for n in notes), default=0) + 1, "owner": owner, "title": title,
                       "body": note.body, "done": False, "created_at": now, "updated_at": now}
            notes.append(created)
            return copy.deepcopy(created)

    @api.get("/api/notes/{note_id:int}", name="get_note")
    def get_note(self, note_id: int, ctx: ApiContext) -> dict:
        with self._lock:
            return copy.deepcopy(self._note(note_id, ctx))

    @api.patch("/api/notes/{note_id:int}", name="edit_note")
    def edit_note(self, note_id: int, edit: NoteEdit, ctx: ApiContext) -> dict:
        """Change a note's title, body or done, and its ``updated_at``."""
        fields = edit.model_dump(exclude_none=True)
        if "title" in fields:
            fields["title"] = _title(fields["title"])
        now = iso(self.now())
        with self._lock:
            note = self._note(note_id, ctx)
            note.update(fields, updated_at=now)
            return copy.deepcopy(note)

    @api.post("/api/notes/{note_id:int}/toggle", name="toggle_note")
    def toggle_note(self, note_id: int, ctx: ApiContext) -> dict:
        """Mark a note done, or not done when it was."""
        now = iso(self.now())
        with self._lock:
            note = self._note(note_id, ctx)
            note.update(done=not note["done"], updated_at=now)
            return copy.deepcopy(note)

    @api.delete("/api/notes/{note_id:int}", name="delete_note")
    def delete_note(self, note_id: int, ctx: ApiContext) -> dict:
        with self._lock:
            note = self._note(note_id, ctx)
            self.data[NOTES] = [n for n in self.data[NOTES] if n is not note]
        return {"deleted": note_id}

    @api.get("/api/admin/notes", name="all_notes", roles=["admin"])
    def all_notes(self) -> list:
        """Every user's notes, for an ``admin`` caller only."""
        with self._lock:
            return copy.deepcopy(self.data.get(NOTES, []))

    def _user(self, ctx: ApiContext) -> str:
        if ctx.user is None:
            raise Unauthorized("sign in first")
        return ctx.user

    def _note(self, note_id: int, ctx: ApiContext) -> dict:
        """The signed-in user's note ``note_id``, read under the lock; another user's is as missing as one that
        doesn't exist."""
        note = next((n for n in self.data.get(NOTES, []) if n["id"] == note_id), None)
        if note is None or note["owner"] != self._user(ctx):
            raise NotFound(f"no note {note_id}")
        return note


def _part_data(part) -> dict:
    """A part's JSON: a data part's data, a text part's text, or a file part's bytes or ``file://`` file."""
    if isinstance(part, DataPart):
        return part.data
    if isinstance(part, TextPart):
        return json.loads(part.text)
    if isinstance(part, FilePart):
        if getattr(part.file, "bytes", None) is not None:
            return json.loads(base64.b64decode(part.file.bytes))
        uri = urlparse(part.file.uri)
        if uri.scheme == "file":
            return json.loads(Path(unquote(uri.path)).read_text(encoding="utf-8"))
        raise ValueError(f"Notes reads a file part by its bytes or a file:// URI, not {part.file.uri!r}")
    raise ValueError(f"Notes can't read a {type(part).__name__}")


def _title(title: str) -> str:
    if not title.strip():
        raise BadRequest("a note needs a title", code="empty_title")
    return title.strip()


if __name__ == "__main__":
    Notes().serve()
